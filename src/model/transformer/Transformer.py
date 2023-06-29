import copy
from pandas import MultiIndex

import torch
from torch import nn
import torch.nn.functional as F


class Transformer(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def encode(self, x):
        out = self.encoder(x)
        return out
    
    def decode(self, z, c):
        out = self.decoder(z, c)
        return out
    
    def forward(self, x, z):
        c = self.encode(x)
        y = self.decode(z, c)
        return y
    
    
class Encoder(nn.Module):
    
    def __init__(self, encoder_block, n_layer):
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
    
    
class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        
    def forward(self, x):
        out = x
        out = self.self_attention(out)
        out = self.position_ff(out)
        return out
    
    
class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h 
        self.qkv_fc = qkv_fc    # weight shape : (d_embed, d_model)
        self.out_fc = out_fc    # weight shape : (d_model, d_embed)
    
    def forward(self, *args, query, key, value, pad_mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # pad_mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc):  # (n_batch, seq_len, d_embed)
            out = fc(x)        # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query, self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc)     # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        out = self.calculate_self_attention(query, key, value, pad_mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        return out
    
    # Transformer의 input은 mini-batch 형태
    # Encoder의 input shape : (n_batch, seq_len, dim_embed)
    # Q, K, V의 shape : (n_batch, seq_len, dim_key), dim_key의 경우 QKV를 만드는 FC-layer에 의해서 결정 됨
    def calculate_self_attention(self, query, key, value, pad_mask):
        # query, key, value: (n_batch, seq_len, d_k)
        # pad_mask: (n_batch, seq_len, seq_len)
        dim_key = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(dim_key)
        if pad_mask is not None:
            attention_score = attention_score.masked_fill(pad_mask==0, -1e9)
        attention_prob = F.soft_max(attention_score, dim=-1)
        out = torch.matmul(attention_prob, value)
        return out