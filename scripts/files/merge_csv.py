import pandas as pd
import os

# Paths to your CSV files
csv_file_path1 = '/home/dino/Desktop/SP24/scripts/Tw_2024_2.csv'
csv_file_path2 = '/home/dino/Desktop/SP24/scripts/Tw_2024_3.csv'
merged_csv_file_path = '/home/dino/Desktop/SP24/scripts/biden_new_2024_2_notformat.csv'

# Chunk size (number of rows processed at a time)
chunk_size = 10000

# Function to merge CSV files
def merge_csv_files(file_path1, file_path2, merged_file_path, chunk_size=10000):
    try:
        # Check column consistency
        cols1 = pd.read_csv(file_path1, nrows=0).columns
        cols2 = pd.read_csv(file_path2, nrows=0).columns
        if not cols1.equals(cols2):
            raise ValueError("CSV files have different columns")

        # Read and append the first file
        print(f"Processing {file_path1}...")
        reader = pd.read_csv(file_path1, chunksize=chunk_size, low_memory=False)
        for chunk in reader:
            chunk.to_csv(merged_file_path, mode='a', index=False, header=not os.path.exists(merged_file_path))
            print(f"Processed {chunk.shape[0]} rows from {file_path1}")

        # Read and append the second file
        print(f"Processing {file_path2}...")
        reader = pd.read_csv(file_path2, chunksize=chunk_size, low_memory=False)
        for chunk in reader:
            chunk.to_csv(merged_file_path, mode='a', index=False, header=not os.path.exists(merged_file_path))
            print(f"Processed {chunk.shape[0]} rows from {file_path2}")

        print(f"Files merged successfully into {merged_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Merge the CSV files
merge_csv_files(csv_file_path1, csv_file_path2, merged_csv_file_path, chunk_size)
