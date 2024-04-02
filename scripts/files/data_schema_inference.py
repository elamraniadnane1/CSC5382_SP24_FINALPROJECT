import pandas as pd
import tensorflow_data_validation as tfdv

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def infer_schema(dataframe):
    """
    Infer schema of the DataFrame using TensorFlow Data Validation.

    Args:
        dataframe (pd.DataFrame): DataFrame for which to infer the schema.

    Returns:
        Schema: Inferred schema.
    """
    stats = tfdv.generate_statistics_from_dataframe(dataframe)
    schema = tfdv.infer_schema(statistics=stats)
    return schema

def save_schema(schema, output_path):
    """
    Save the inferred schema to a file.

    Args:
        schema (Schema): Inferred schema.
        output_path (str): File path to save the schema.
    """
    with open(output_path, 'w') as f:
        f.write(str(schema))

def main():
    input_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\data.csv'
    output_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\data_schema.pbtxt'

    # Load data
    data = load_data(input_file_path)

    # Infer schema
    schema = infer_schema(data)

    # Save schema
    save_schema(schema, output_file_path)
    print(f"Schema saved to {output_file_path}")

if __name__ == "__main__":
    main()
