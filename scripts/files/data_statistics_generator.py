import tensorflow_data_validation as tfdv
import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def generate_statistics(dataframe):
    """
    Generate statistics using TensorFlow Data Validation for the given DataFrame.

    Args:
        dataframe (pd.DataFrame): DataFrame for which to generate statistics.

    Returns:
        tfdv.StatsOptions: Generated statistics for the DataFrame.
    """
    return tfdv.generate_statistics_from_dataframe(dataframe)

def main():
    file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\data.csv'
    data = load_data(file_path)

    # Generate statistics
    stats = generate_statistics(data)

    # Visualize statistics
    tfdv.visualize_statistics(stats)

if __name__ == "__main__":
    main()
