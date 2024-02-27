import os
import subprocess

def is_ceph_cluster_bootstrapped():
    """ Check if a Ceph cluster is bootstrapped by looking for config files. """
    return os.path.exists('/etc/ceph/ceph.conf')

def is_keyring_available():
    """ Check if the keyring files are available. """
    keyring_files = [
        '/etc/ceph/ceph.client.admin.keyring',
        '/etc/ceph/ceph.keyring',
        '/etc/ceph/keyring',
        '/etc/ceph/keyring.bin'
    ]
    return any(os.path.exists(keyring) for keyring in keyring_files)

def recover_or_generate_keyring():
    """
    Attempt to recover or generate the keyring files.
    This is a placeholder and should be replaced with actual commands specific to your Ceph setup.
    """
    try:
        # Example command to generate a new admin keyring
        # This is just an example and might not directly apply to your setup
        subprocess.run(["sudo", "ceph-authtool", "/etc/ceph/ceph.client.admin.keyring", "--create-keyring", "--gen-key", "-n", "client.admin", "--set-uid=0", "--cap", "mon", "'allow *'", "--cap", "osd", "'allow *'", "--cap", "mds", "'allow'"], check=True)
        print("Keyring files recovered or generated.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to recover or generate keyring: {e}")

def enable_ceph_cli():
    """ Enable Ceph CLI by running the necessary commands. """
    try:
        # Check if ceph-common is already installed
        ceph_common_installed = subprocess.run(["sudo", "ceph", "-v"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if ceph_common_installed.returncode != 0:
            # Install ceph-common package
            subprocess.run(["sudo", "cephadm", "add-repo", "--release", "reef"], check=True)
            subprocess.run(["sudo", "cephadm", "install", "ceph-common"], check=True)
        else:
            print("ceph-common is already installed.")

        # Verify ceph command
        subprocess.run(["sudo", "ceph", "-v"], check=True)

        if not is_keyring_available():
            print("Keyring files are missing. Attempting to recover or generate keyring files.")
            recover_or_generate_keyring()

        if is_keyring_available():
            # Run ceph status only if keyring files are present
            subprocess.run(["sudo", "ceph", "status"], check=True)
            print("Ceph CLI is enabled and functional.")
        else:
            print("Keyring files are still missing. Please ensure that the keyring files are in place and accessible.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if is_ceph_cluster_bootstrapped():
        enable_ceph_cli()
    else:
        print("No bootstrapped Ceph cluster found.")

