import subprocess
import os

def run_command(command):
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
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

def main():
    repo_path = "/home/dino/Documents/SP24"
    
    # Initialize Git LFS
    if not run_command("git lfs install"):
        return

    # Change directory to the repo
    os.chdir(repo_path)
    
    # Find and track large files
    large_files = find_large_files(repo_path)
    for file in large_files:
        if not run_command(f"git lfs track '{file}'"):
            return
    
    # Add all changes including large files
    if not run_command("git add ."):
        return
    
    # Commit changes
    commit_message = "Update with large file handling"
    if not run_command(f"git commit -m '{commit_message}'"):
        return
    
    # Push changes to the remote repository
    if not run_command("git push"):
        return

    print("Changes have been successfully pushed to the repository.")

if __name__ == "__main__":
    main()
