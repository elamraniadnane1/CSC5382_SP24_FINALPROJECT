# File: load_data.py

import pandas as pd
from typing import Dict

def load_data(file_path: str) -> Dict:
    """
    Loads data from a CSV file and returns it as a dictionary.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        Dict: A dictionary containing the loaded data.
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Optional: Display the first few rows of the dataframe
        print("Sample data from the loaded dataset:")
        print(df.head())

        # Convert the dataframe to a dictionary
        data_dict = df.to_dict(orient='list')

        return data_dict

    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found at {file_path}")
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")
