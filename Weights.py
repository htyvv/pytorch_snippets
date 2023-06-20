def load_weights(
    pretrained_checkpoint_path: str, 
    target_model: torch.nn.Module, 
    only_rnn: bool=True
) -> torch.nn.Module:  
    target_state_dict = target_model.state_dict()
    pretrained_state_dict = torch.load(pretrained_checkpoint_path)['state_dict']
    if only_rnn:
        rnn_state_dict = load_rnn_weights(pretrained_state_dict)
        check_inclusive_keys(target_state_dict, rnn_state_dict)
        target_state_dict.update(rnn_state_dict)
    target_model.load_state_dict(target_state_dict)
    return target_model

def load_rnn_weights(
    pretrained_state_dict
):
    rnn_state_dict = {
        weight_name: weight
        for weight_name, weight in pretrained_state_dict.items()
        if 'rnn' in weight_name
    }
    return rnn_state_dict

def check_inclusive_keys(
    main_dict, sub_dict
):
    for key in sub_dict:
        if main_dict.get(key) == None:
            raise "Two dictionary keys do not matched"
