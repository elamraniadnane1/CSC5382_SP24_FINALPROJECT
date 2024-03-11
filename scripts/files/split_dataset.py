import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(df, test_size, val_size):
    """
    Splits the dataset into training, validation, and testing subsets.

    Args:
    df (pd.DataFrame): The full dataset.
    test_size (float): Proportion of the dataset to be used as the test set.
    val_size (float): Proportion of the training dataset to be used as the validation set.

    Returns:
    dict: A dictionary containing the training, validation, and test DataFrames.
    """

    # Split the data into training and testing
    train_df, test_df = train_test_split(df, test_size=test_size)

    # Further split the training data into training and validation
    train_size = 1.0 - val_size
    train_df, val_df = train_test_split(train_df, test_size=val_size / train_size)

    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }

def main():
    # Load your dataset here
    # Example: df = pd.read_csv('path_to_your_dataset.csv')
    
    # Define the test and validation size
    test_size = 0.2  # 20% of the dataset for testing
    val_size = 0.1   # 10% of the training dataset for validation

    # Split the dataset
    split_data = split_dataset(df, test_size, val_size)

    # Save the datasets to disk or pass them on to the next step
    # Example:
    # split_data['train_df'].to_csv('train_dataset.csv', index=False)
    # split_data['val_df'].to_csv('validation_dataset.csv', index=False)
    # split_data['test_df'].to_csv('test_dataset.csv', index=False)

if __name__ == "__main__":
    main()
