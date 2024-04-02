import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def partition_data(df, test_size=0.2, val_size=0.1):
    """
    Partition the data into training, validation, and testing sets.

    Args:
        df (pd.DataFrame): DataFrame to partition.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.

    Returns:
        tuple: (train_data, val_data, test_data) as DataFrames.
    """
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
    val_size_adjusted = val_size / (1 - test_size)
    train_data, val_data = train_test_split(train_data, test_size=val_size_adjusted, random_state=42)
    return train_data, val_data, test_data

def save_data(df, file_path):
    """
    Save DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the CSV file.
    """
    df.to_csv(file_path, index=False)

def main():
    input_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\dataset.csv'
    train_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\train_data.csv'
    val_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\val_data.csv'
    test_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\test_data.csv'

    # Load data
    data = load_data(input_file_path)

    # Partition data
    train_data, val_data, test_data = partition_data(data)

    # Save partitioned data
    save_data(train_data, train_file_path)
    save_data(val_data, val_file_path)
    save_data(test_data, test_file_path)
    print(f"Partitioned data saved to {train_file_path}, {val_file_path}, and {test_file_path}")

if __name__ == "__main__":
    main()
