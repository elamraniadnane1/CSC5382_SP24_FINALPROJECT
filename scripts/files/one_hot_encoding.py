import pandas as pd

def one_hot_encode(df, columns_to_encode):
    """
    Apply one-hot encoding to the specified columns in the DataFrame.

    :param df: The DataFrame containing the data.
    :param columns_to_encode: List of column names to apply one-hot encoding.
    :return: DataFrame with one-hot encoded columns.
    """
    return pd.get_dummies(df, columns=columns_to_encode, prefix=columns_to_encode)

# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = {
        'Color': ['Red', 'Blue', 'Green', 'Blue'],
        'Size': ['Small', 'Medium', 'Large', 'Medium']
    }
    df = pd.DataFrame(sample_data)

    # Columns to apply one-hot encoding
    columns_to_encode = ['Color', 'Size']

    # Apply one-hot encoding
    encoded_df = one_hot_encode(df, columns_to_encode)
    print("One-Hot Encoded DataFrame:\n", encoded_df)
