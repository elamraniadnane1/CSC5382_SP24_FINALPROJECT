import os

def ignore_large_files(directory, size_limit_mb):
    """
    List files in a directory, excluding those larger than a specified size.

    :param directory: Path to the directory.
    :param size_limit_mb: File size limit in megabytes.
    :return: List of file paths under the size limit.
    """
    files_under_limit = []

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert size to MB
            if size_mb <= size_limit_mb:
                files_under_limit.append(file_path)

    return files_under_limit

if __name__ == "__main__":
    directory_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files'
    size_limit = 50  # Size limit in MB

    valid_files = ignore_large_files(directory_path, size_limit)
    print("Files under the size limit:")
    for file in valid_files:
        print(file)
