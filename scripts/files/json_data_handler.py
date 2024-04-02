import json
import pandas as pd

def load_json(file_path):
    """
    Load JSON data from a file.

    :param file_path: Path to the JSON file.
    :return: Data loaded from the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def process_json_data(data):
    """
    Process JSON data (modify this function based on your processing needs).

    :param data: JSON data to be processed.
    :return: Processed data.
    """
    # Example processing: converting JSON to DataFrame
    # This can be customized based on the structure of your JSON data
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    return df

def save_json(data, output_file_path):
    """
    Save data in JSON format to a file.

    :param data: Data to be saved.
    :param output_file_path: File path to save the JSON data.
    """
    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

if __name__ == "__main__":
    # Example usage
    input_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\input.json'
    output_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\output.json'

    # Load JSON data
    json_data = load_json(input_path)

    # Process the data
    processed_data = process_json_data(json_data)

    # Save the processed data back to JSON
    save_json(processed_data.to_dict(orient='records'), output_path)
