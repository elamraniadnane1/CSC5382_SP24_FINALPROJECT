import pandas as pd

def load_data(csv_file_path: str) -> dict:
    """
    Load data from a CSV file.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        dict: Loaded data in dictionary format.
    """
    data = pd.read_csv(csv_file_path)
    # Assuming data preprocessing or formatting if necessary
    # Example:
    # processed_data = preprocess_data(data)
    return data.to_dict()
