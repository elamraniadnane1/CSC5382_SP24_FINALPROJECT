import subprocess
import json
import os
import shutil

def run_command(command, return_output=False):
    try:
        result = subprocess.run(command, check=True, capture_output=return_output, text=True)
        if return_output:
            return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return None
    return True

def is_ceph_command_available():
    return shutil.which("ceph") is not None

def check_ceph_permissions():
    try:
        subprocess.run(["ceph", "-s"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Permission denied or Ceph cluster not accessible: {e}")
        return False
    return True

def get_cluster_status():
    status_output = run_command(["ceph", "status", "--format=json"], return_output=True)
    if status_output:
        try:
            return json.loads(status_output)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    else:
        print("Failed to get cluster status or empty output.")
    return None
def get_monitors(status):
    try:
        if 'quorum_names' in status:
            return status['quorum_names']
        else:
            print("Monitor information not found in 'quorum_names'.")
    except KeyError as e:
        print(f"KeyError encountered when accessing monitor information: {e}")
    return []


def detect_network_and_nodes():
    status = get_cluster_status()
    if status:
        monitors = get_monitors(status)
        public_network = get_public_network(status)
        if not public_network:
            print("No public network information found in cluster status.")
            return None, None

        if not monitors:
            print("No monitors detected in the cluster. Preparing to deploy a new monitor.")
            return public_network, None

        if 'osdmap' in status and 'osds' in status['osdmap']:
            nodes = list(set([osd['hostname'] for osd in status['osdmap']['osds']]))
            return public_network, nodes

        print("No OSD information found in cluster status.")
        return public_network, None

    print("Failed to retrieve cluster status.")
    return None, None


def deploy_monitors(nodes, network):
    if not nodes or not network:
        print("No nodes or network detected. Exiting.")
        return False

    if not set_public_network(network):
        return False

    if not run_command(["ceph", "orch", "apply", "mon", "--unmanaged"]):
        return False

    for node in nodes:
        if not run_command(["ceph", "orch", "daemon", "add", f"mon {node}:{network}"]):
            return False

    placement = ",".join(nodes)
    return run_command(["ceph", "orch", "apply", "mon", f"--placement={placement}"])

# Check if the script is run as root
if os.geteuid() != 0:
    print("This script must be run with sudo privileges.")
    exit(1)

# Main
if not is_ceph_command_available():
    print("Ceph command not found. Please ensure Ceph is installed and the command is in your PATH.")
elif not check_ceph_permissions():
    print("Failed to access Ceph cluster. Please check your permissions.")
else:
    network, nodes = detect_network_and_nodes()
    if network:
        if deploy_monitors(network, nodes):
            print("Monitors deployed successfully.")
        else:
            print("Failed to deploy monitors.")
