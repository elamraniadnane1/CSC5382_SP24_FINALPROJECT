import zipfile
import os

# Function to unzip a file
def unzip_file(zip_file_path, extraction_directory):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_directory)
    print(f"Unzipped '{zip_file_path}' to '{extraction_directory}'")

# Define the path to the zip file (assuming it's in the same directory as the script)
zip_file_path = 'kawintiranon-stance-detection.zip'

# Get the directory where the script is running
script_directory = os.path.dirname(os.path.realpath(__file__))

# Unzip the file in the script's directory
unzip_file(zip_file_path, script_directory)