import subprocess
import os
from collections import defaultdict

def run_command(command):
    try:
        subprocess.check_call(command, shell=False)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    except OSError as e:
        print(f"OS Error: {e}")
        return False
    return True

def find_large_files(directory, size_threshold=100 * 1024 * 1024):  # 100 MB
    large_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) > size_threshold:
                large_files.append(file_path)
    return large_files

def get_extensions_of_large_files(large_files):
    extensions = defaultdict(bool)
    for file in large_files:
        extension = os.path.splitext(file)[1]
        if extension:  # Ignore files without an extension
            extensions[extension] = True
    return extensions

def has_changes_to_commit():
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"]).strip().decode('utf-8')
        return len(status) != 0
    except subprocess.CalledProcessError as e:
        print(f"Error checking git status: {e}")
        return False

def main():
    repo_path = "/home/dino/Desktop/SP24"
    
    # Initialize Git LFS
    if not run_command(["git", "lfs", "install"]):
        return

    # Change directory to the repo
    os.chdir(repo_path)
    
    # Find large files
    large_files = find_large_files(repo_path)
    extensions = get_extensions_of_large_files(large_files)

    # Track large files by extension
    for ext in extensions:
        if not run_command(["git", "lfs", "track", f"*{ext}"]):
            return

    # Re-add all files to ensure they are correctly tracked by LFS
    if not run_command(["git", "add", "--renormalize", "."]):
        return

    # Add all changes including large files
    if not run_command(["git", "add", "."]):
        return
    
    # Check if there are changes to commit
    if has_changes_to_commit():
        # Commit changes
        commit_message = "Update with large file handling"
        if not run_command(["git", "commit", "-m", commit_message]):
            return
    else:
        print("No changes to commit.")
    
    # Push changes to the remote repository
    if not run_command(["git", "push"]):
        return

    print("Changes have been successfully pushed to the repository.")

if __name__ == "__main__":
    main()
