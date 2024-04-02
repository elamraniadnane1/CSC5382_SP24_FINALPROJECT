import pandas as pd

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def calculate_statistics(df):
    """
    Calculate basic statistics for a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to calculate statistics for.

    Returns:
        pd.DataFrame: Statistics for each column.
    """
    stats = df.describe().transpose()
    return stats

def save_statistics(stats, output_path):
    """
    Save the statistics DataFrame to a CSV file.

    Args:
        stats (pd.DataFrame): Statistics DataFrame.
        output_path (str): File path to save the statistics.
    """
    stats.to_csv(output_path)

def main():
    input_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\data.csv'
    output_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\data_statistics.csv'

    # Load data
    data = load_data(input_file_path)

    # Calculate statistics
    statistics = calculate_statistics(data)

    # Save statistics
    save_statistics(statistics, output_file_path)
    print(f"Statistics saved to {output_file_path}")

if __name__ == "__main__":
    main()
