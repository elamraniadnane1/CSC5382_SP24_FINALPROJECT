import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def convert_categorical_to_numerical(df, categorical_columns):
    """
    Convert categorical columns in a DataFrame to numerical format.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        categorical_columns (list): List of column names that are categorical.

    Returns:
        pd.DataFrame: DataFrame with categorical columns converted to numerical.
    """
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le  # Store the label encoder for each column

    return df, label_encoders

def main():
    file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\dataset.csv'
    output_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\numerical_data.csv'

    # Load data
    data = load_data(file_path)

    # List of categorical columns to convert
    categorical_columns = ['category1', 'category2']  # Update with your actual column names

    # Convert and get label encoders
    numerical_data, encoders = convert_categorical_to_numerical(data, categorical_columns)

    # Save the converted data
    numerical_data.to_csv(output_file_path, index=False)
    print(f"Converted data saved to {output_file_path}")

if __name__ == "__main__":
    main()
