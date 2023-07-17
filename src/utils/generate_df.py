import numpy as np
import pandas as pd


def generate_df(df_size: float, print_option=False):
    """
    df_size : 원하는 dataframe 용량 (MB)
    """
    # Define the desired size of the DataFrame in bytes
    target_size = int(df_size * 1024 * 1024)  # MB

    # Initialize an empty DataFrame
    df = pd.DataFrame()

    # Define the number of rows to generate
    num_rows = target_size // 8  # Assuming each value takes 8 bytes (64-bit float)

    # Generate random data for each column
    for i in range(10):  # Generate 10 columns
        column_name = f'column_{i}'
        df[column_name] = np.random.randn(num_rows)

    # Check the size of the DataFrame in bytes
    df_size = df.memory_usage(index=True).sum()

    # If the generated DataFrame is smaller than the target size, repeat the rows
    while df_size < target_size:
        df = pd.concat([df, df.sample(n=num_rows, replace=True)])
        df_size = df.memory_usage(index=True).sum()

    # Reset the index of the DataFrame
    df.reset_index(drop=True, inplace=True)

    # Check the final size of the DataFrame
    if print_option:
        print(f"✓ Final DataFrame size: {df_size / 1024 / 1024:.2f} MB")
        print(f"✓ Final DataFrame shape: {df.shape}")
    
    return df