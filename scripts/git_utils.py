import os
import subprocess

# Function to get all large files in a directory
def get_large_files(directory, threshold):
    large_files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.isfile(filepath) and os.path.getsize(filepath) > threshold:
                large_files.append(filepath)
    return large_files

# Function to push changes to GitHub
def git_push_commit(directory, message):
    try:
        # Add all files
        subprocess.check_output(['git', 'add', '.'], cwd=directory)
        
        # Commit changes
        subprocess.check_output(['git', 'commit', '-m', message], cwd=directory)
        
        # Push to GitHub
        subprocess.check_output(['git', 'push'], cwd=directory)
        
        print("Pushed changes to GitHub successfully.")
    except subprocess.CalledProcessError as e:
        print("Error:", e.output)

# Directory to search for large files
directory = '/home/dino/Desktop/SP24/'

# Threshold for large files in bytes (adjust as needed)
threshold = 10 * 1024 * 1024  # 10 MB

# Message for commit
commit_message = "Added large files"

# Get large files
large_files = get_large_files(directory, threshold)

# If there are large files, push and commit
if large_files:
    print("Found large files:")
    for file in large_files:
        print(file)
    git_push_commit(directory, commit_message)
else:
    print("No large files found.")
