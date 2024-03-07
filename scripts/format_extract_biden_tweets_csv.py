import pandas as pd

# File paths
input_csv_file = '/home/dino/Desktop/SP24/scripts/biden_new_2024_2_notformat.csv'  # Replace with your input file path
output_csv_file = '/home/dino/Desktop/SP24/scripts/biden_new_2024_2.csv'  # Replace with your output file path

def check_column_consistency(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

def convert_csv_format(input_file, output_file):
    try:
        # Read the input CSV file
        df = pd.read_csv(input_file)

        # Check if the required columns are present
        check_column_consistency(df, ['id', 'tweetText'])

        # Select and rename the necessary columns
        df = df[['id', 'tweetText']]
        df.columns = ['tweet_id', 'text']

        # Add a new column 'label' with default value 'NONE'
        df['label'] = 'NONE'

        # Save the modified DataFrame to a new CSV file
        df.to_csv(output_file, index=False)
        print("File converted and saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Convert the CSV file
convert_csv_format(input_csv_file, output_csv_file)
