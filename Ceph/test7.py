import subprocess
import re
def run_command(command, return_output=False):
    try:
        result = subprocess.run(command, check=True, shell=True, text=True, capture_output=return_output)
        if return_output:
            return result.stdout.strip()
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return False
    
def get_non_os_partitions():
    partitions_output = run_command("lsblk -no MOUNTPOINT,NAME", True)
    non_os_partitions = [line.split()[1] for line in partitions_output.splitlines() if line.split()[0] == '']
    return non_os_partitions

def get_physical_device_for_lv(lv_path):
    cmd = f"lvdisplay {lv_path} | grep 'LV Path' | awk '{{print $3}}'"
    physical_device = run_command(cmd, True)
    return physical_device.split('/')[-1]

def is_partition_used_for_osd(device):
    osd_partitions = list_osd_partitions()
    return f"/dev/{device}" in osd_partitions

def print_osd_details():
    osds_output = run_command("sudo ceph-volume lvm list", True)
    osd_id = 0
    for line in osds_output.splitlines():
        if line.startswith("======"):
            osd_id = re.findall(r"\d+", line)[0]
        if 'devices' in line:
            device = line.split()[-1]
            if device.startswith("/dev"):
                # Handle logical volumes
                if '/dev/ceph-' in device:
                    device = get_physical_device_for_lv(device)
                print(f"====== osd.{osd_id} =======")
                print(f"  [block]       {device}\n")
                print(f"      devices                   {device}")

def clean_filesystem(device):
    confirmation = input(f"Are you sure you want to wipe all filesystems on {device}? [y/N]: ")
    if confirmation.lower() != 'y':
        print(f"Skipping wipe on {device}")
        return True
    return run_command(f"sudo wipefs --all {device}")

def deploy_osd():
    non_os_partitions = get_non_os_partitions()
    if not non_os_partitions:
        print("No non-OS partitions found.")
        return

    for device in non_os_partitions:
        print(f"Selected partition: /dev/{device}")

        if is_partition_used_for_osd(device):
            print(f"/dev/{device} is already used for an OSD.")
            continue

        # Cleaning filesystem
        if not clean_filesystem(f"/dev/{device}"):
            print(f"Failed to clean filesystem on /dev/{device}.")
            continue

        # Check for the existence and permissions of the bootstrap OSD keyring
        if not run_command("sudo ls /var/lib/ceph/bootstrap-osd/ceph.keyring"):
            print("Bootstrap OSD keyring does not exist.")
            continue

        if not run_command("sudo ls -l /var/lib/ceph/bootstrap-osd/ceph.keyring"):
            print("Unable to verify permissions of the bootstrap OSD keyring.")
            continue

        # Reset bootstrap OSD permissions
        if not run_command(
            "sudo ceph auth get-or-create client.bootstrap-osd mon 'allow profile bootstrap-osd' -o /var/lib/ceph/bootstrap-osd/ceph.keyring"
        ):
            print("Failed to reset bootstrap OSD permissions.")
            continue

        # Deploy OSD
        if not run_command(f"sudo ceph-volume lvm create --data /dev/{device}"):
            print(f"Failed to deploy OSD on /dev/{device}.")
            continue

        # Check Ceph cluster health
        if not run_command("sudo ceph -s"):
            print("Failed to check Ceph cluster health.")
            continue

        print(f"OSD deployed successfully on /dev/{device}.")

def list_osd_partitions():
    osds_output = run_command("sudo ceph-volume lvm list", True)
    osd_partitions = []
    for line in osds_output.splitlines():
        if 'devices' in line:
            device = line.split()[-1]
            if device.startswith("/dev"):
                # Handle logical volumes
                if '/dev/ceph-' in device:
                    device = get_physical_device_for_lv(device)
                osd_partitions.append(device)
    return osd_partitions
    return f"/dev/{device}" in osd_partitions


# Example usage
deploy_osd()
print_osd_details()