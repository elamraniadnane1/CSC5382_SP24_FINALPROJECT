import subprocess
import re
import socket
import os
import json


def execute_command(command, print_output=False):
    """Executes a given command in the shell and returns the output."""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if print_output:
            print(result.stdout)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stderr:
            print(e.stderr)
        return False, None

def get_cluster_fsids():
    """Gets the fsid of all existing clusters."""
    list_command = "sudo cephadm ls"
    success, output = execute_command(list_command)
    if success and output:
        fsids = re.findall(r'\"fsid\": \"([0-9a-f-]+)\"', output)
        return list(set(fsids))  # Remove duplicates
    return []

def prepare_partition(partition):
    """Formats and prepares a partition for OSD deployment."""
    print(f"Preparing {partition} for OSD deployment...")
    # Format the partition (replace ext4 with your desired filesystem)
    format_command = f"sudo mkfs.ext4 {partition}"
    execute_command(format_command)

    # Wipe any existing filesystem signatures
    wipe_command = f"sudo wipefs --all {partition}"
    execute_command(wipe_command)

def delete_clusters():
    """Deletes all Ceph clusters based on their fsid."""
    fsids = get_cluster_fsids()
    for fsid in fsids:
        print(f"Deleting cluster with fsid: {fsid}")
        delete_command = f"sudo cephadm rm-cluster --force --fsid {fsid}"
        execute_command(delete_command)

def get_host_ip():
    """Gets the IP address of the host."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def bootstrap_cluster():
    """Bootstraps a new Ceph cluster."""
    mon_ip = get_host_ip()
    print(f"Bootstrapping new cluster with monitor IP: {mon_ip}")
    bootstrap_command = f"sudo cephadm bootstrap --mon-ip {mon_ip} --allow-overwrite"
    execute_command(bootstrap_command, print_output=True)

def check_dependency(command, package_name):
    """Check if a dependency is installed and print its version."""
    version_command = f"{command} --version"
    try:
        # Some commands might output the version info to stderr instead of stdout
        result = subprocess.run(version_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.stdout:
            version_info = result.stdout.strip().split('\n')[0]  # Get the first line of output
        else:
            version_info = "Version info not available"
        print(f"{package_name} version: {version_info}")
        return True
    except subprocess.CalledProcessError:
        print(f"{package_name} is not installed.")
        return False


def check_python():
    return check_dependency("python3", "Python 3")

def check_systemd():
    return check_dependency("systemctl", "Systemd")

def check_container_runtime():
    if check_dependency("podman", "Podman"):
        return True
    elif check_dependency("docker", "Docker"):
        return True
    else:
        print("Neither Podman nor Docker is installed.")
        return False

def check_time_sync():
    if check_dependency("chronyc", "Chrony"):
        return True
    elif check_dependency("ntpd", "NTP"):
        return True
    else:
        print("Neither Chrony nor NTP is installed.")
        return False

def check_lvm2():
    return check_dependency("lvm", "LVM2")

def check_ssh():
    return check_dependency("ssh", "SSH")

def install_package(package_name):
    """Installs a given package using the system's package manager."""
    # Determine the package manager
    if os.path.exists("/usr/bin/apt"):
        install_command = f"sudo apt-get install -y {package_name}"
    elif os.path.exists("/usr/bin/yum"):
        install_command = f"sudo yum install -y {package_name}"
    elif os.path.exists("/usr/bin/dnf"):
        install_command = f"sudo dnf install -y {package_name}"
    else:
        print("Unsupported package manager.")
        return False

    # Execute the installation command
    print(f"Installing {package_name}...")
    success, _ = execute_command(install_command)
    if success:
        print(f"{package_name} installed successfully.")
    else:
        print(f"Failed to install {package_name}.")
    return success

def ensure_dependency(command, name, package=None):
    if package is None:
        package = command
    try:
        result = subprocess.run([command, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print(f"{name} is installed.")
            return True
        else:
            print(f"{name} is not installed. Attempting to install...")
            install_result = subprocess.run(["sudo", "apt-get", "install", "-y", package], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return install_result.returncode == 0
    except FileNotFoundError:
        print(f"{name} is not installed. Attempting to install...")
        install_result = subprocess.run(["sudo", "apt-get", "install", "-y", package], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return install_result.returncode == 0


def check_ceph():
    """Check if Ceph is installed and print its version."""
    ceph_version_command = "ceph --version"
    try:
        result = subprocess.run(ceph_version_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.stdout:
            version_info = result.stdout.strip().split('\n')[0]
            print(f"Ceph version: {version_info}")
            return True
        else:
            print("Ceph version info not available")
            return False
    except subprocess.CalledProcessError:
        print("Ceph is not installed.")
        return False

def ensure_ceph():
    """Ensure Ceph and ceph-volume are installed, installing them if necessary."""
    if not check_ceph():
        if not install_package("ceph"):
            return False
    # Check for ceph-volume specifically
    if not check_dependency("ceph-volume", "Ceph Volume"):
        print("Attempting to install ceph-volume separately...")
        return install_package("ceph-volume")
    return True
def ensure_ceph_volume():
    """Ensure ceph-volume is installed."""
    ceph_volume_command = "ceph-volume --version"
    try:
        result = subprocess.run(ceph_volume_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if result.stdout:
            version_info = result.stdout.strip().split('\n')[0]
            print(f"Ceph-volume version: {version_info}")
            return True
        else:
            print("Ceph-volume version info not available")
            return False
    except subprocess.CalledProcessError:
        print("Ceph-volume is not installed.")
        return False
    
def check_all_dependencies():
    print("Checking and installing dependencies...")
    dep_checks = [
        check_python(), check_systemd(), 
        check_container_runtime(), check_time_sync(), 
        check_lvm2(), check_ssh(), ensure_ceph(), ensure_ceph_volume()
    ]
    if all(dep_checks):
        print("All dependencies are installed and up to date.")
        # Print versions of all dependencies
        print("Installed dependency versions:")
        check_python()
        check_systemd()
        check_container_runtime()  # This will check both Podman and Docker
        check_time_sync()  # This will check both Chrony and NTP
        check_lvm2()
        check_ssh()
        check_ceph()
        ensure_ceph_volume()
    else:
        print("Some dependencies could not be installed.")

def is_ceph_osd_installed():
    """Check if ceph-osd package is installed."""
    check_command = "dpkg -l | grep ceph-osd || rpm -q ceph-osd"
    success, output = execute_command(check_command)
    return bool(output)


def uninstall_ceph_osd():
    """Uninstall ceph-osd package if installed."""
    if is_ceph_osd_installed():
        print("Uninstalling ceph-osd...")
        uninstall_command = "sudo apt-get remove -y ceph-osd || sudo yum remove -y ceph-osd || sudo dnf remove -y ceph-osd"
        success, _ = execute_command(uninstall_command)
        if success:
            print("ceph-osd uninstalled successfully.")
        else:
            print("Failed to uninstall ceph-osd.")
    else:
        print("ceph-osd is not installed.")

def device_meets_criteria(device):
    """Check if a storage device meets the criteria for OSD deployment."""
    # Assuming 'devices' is a list of dictionaries where each dictionary represents a storage device
    for dev in device.get('devices', []):
        if (not dev.get('partitions') and not dev.get('lvms') and not dev.get('mounted') 
            and not dev.get('filesystem') and not dev.get('ceph_daemon') and dev.get('size', 0) > 5*1024**3):
            return True
    return False

def list_storage_devices():
    """List storage devices available for OSD deployment."""
    list_command = "sudo ceph orch device ls --format=json"
    success, output = execute_command(list_command)
    if success and output:
        try:
            hosts = json.loads(output)
            available_devices = []
            for host in hosts:
                # Assuming 'devices' is a list of device dictionaries
                for device in host.get('devices', []):
                    if device_meets_criteria({'devices': [device]}):
                        # Adding hostname to the device dictionary
                        device['hostname'] = host['name']
                        available_devices.append(device)
            return available_devices
        except json.JSONDecodeError:
            print("Failed to decode JSON from the output.")
            return []
    return []




def deploy_osd_on_partition(partition):
    """Deploys an OSD on a specific partition."""
    print(f"Deploying OSD on {partition}...")
    deploy_command = f"sudo ceph-volume lvm create --data {partition}"
    success, _ = execute_command(deploy_command)
    if success:
        print(f"OSD deployed successfully on {partition}.")
    else:
        print(f"Failed to deploy OSD on {partition}. Please check if 'ceph-volume' is installed and the partition meets OSD requirements.")

def deploy_osds_with_fallback(dry_run=False, unmanaged=False, fallback_partition=None):
    """Deploy OSDs on available devices or fallback to a specific partition."""
    available_devices = list_storage_devices()
    if available_devices:
        for device in available_devices:
            deploy_command = f"sudo ceph orch daemon add osd {device['hostname']}:{device['path']}"
            if dry_run:
                deploy_command += " --dry-run"
            if unmanaged:
                deploy_command += " --unmanaged=true"
            success, _ = execute_command(deploy_command)
            if not success:
                print(f"Failed to deploy OSD on {device['path']}")
    elif fallback_partition:
        prepare_partition(fallback_partition)
        deploy_osd_on_partition(fallback_partition)
    else:
        print("No suitable devices found for OSD deployment and no fallback partition specified.")

# Main execution
print("Welcome to the Ceph Cluster Setup Script!")
print("This script will guide you through setting up a Ceph Cluster.")
# Update and upgrade the system
os.system('sudo apt update && sudo apt upgrade -y')

# Install necessary packages
os.system('sudo apt install -y nginx mysql-server php-fpm php-mysql')

# Instructions to open the sudoers file and add a specific user
# Note: The actual editing cannot be automated in the script. 
# Users need to manually add 'dino ALL=(ALL) NOPASSWD: ALL' in the sudoers file.
print("\nPlease open the sudoers file by running 'sudo visudo'")
print("Then add the following line to allow user 'dino' to run all commands without a password:")
print("dino ALL=(ALL) NOPASSWD: ALL\n")

# Restart the services
os.system('sudo systemctl restart nginx')
os.system('sudo systemctl restart mysql')
os.system("sudo systemctl restart php7.4-fpm") # Replace with your PHP version


# Print completion message# Install Ceph
os.system('sudo apt install -y ceph')

# Verify if ceph-volume is available
exit_code = os.system('ceph-volume --help')
if exit_code != 0:
    print("Error: ceph-volume is not available. Please ensure Ceph is installed correctly.")
else:
    print("ceph-volume is available.")
print("First part of installation and setup completed.")
print("Let's start by checking all required dependencies...")

check_all_dependencies()

# Delete existing clusters
delete_clusters()

# Uninstall ceph-osd if necessary
uninstall_ceph_osd()

# Bootstrap new cluster
bootstrap_cluster()

print("Ceph Cluster setup is complete!")

# Optional: Run a dry run of OSD deployment
deploy_osds_with_fallback(dry_run=True, fallback_partition="/dev/sda4")

# Deploy OSDs with unmanaged option or fallback to a specific partition
deploy_osds_with_fallback(unmanaged=True, fallback_partition="/dev/sda4")