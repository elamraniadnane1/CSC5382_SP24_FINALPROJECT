import subprocess
import os

def run_command(command):
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False
    return True

def main():
    repo_path = "/home/dino/Documents/SP24"
    
    # Initialize Git LFS (if not already initialized)
    if not run_command("git lfs install"):
        return
    
    # Change directory to the repo
    os.chdir(repo_path)
    
    # Track large files with Git LFS (specify your file patterns)
    # Example: run_command("git lfs track '*.zip'")
    
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
