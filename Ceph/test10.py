import subprocess
import os
import logging
logging.basicConfig(level=logging.INFO)

def print_progress(step, total_steps, message):
    """Prints the progress as a percentage."""
    progress = (step / total_steps) * 100
    logging.info(f"{message} - Progress: {progress:.2f}%")

def execute_command(command, step, total_steps):
    """Executes a given command and returns the output, also logs progress."""
    try:
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        progress = (step / total_steps) * 100
        logging.info(f"Step {step}/{total_steps} completed - Progress: {progress:.2f}%")
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing command: {e}")
        if e.stderr:
            logging.error(e.stderr.strip())
        return False, None


def create_rbd_pool(pool_name='myrbdpool', step=1, total_steps=5):
    """Creates a new RBD pool."""
    logging.info(f"Creating RBD pool: {pool_name}")
    success, output = execute_command(f"sudo ceph osd pool create {pool_name} 128 128", step, total_steps)
    if not success:
        logging.error(f"Failed to create pool {pool_name}. Check if the pool already exists or there are permission issues.")
    return success

def create_rbd_image(pool_name, image_name, size, step, total_steps):
    """Creates a new RBD image."""
    logging.info(f"Creating RBD image: {image_name} in pool: {pool_name} with size {size}")
    success, output = execute_command(f"sudo rbd create {pool_name}/{image_name} --size {size}", step, total_steps)
    if not success:
        logging.error(f"Failed to create RBD image. Check if the pool is configured correctly, if you have the necessary permissions, and if the size specification is valid.")
    return success



def map_rbd_image(pool_name, image_name):
    """Maps the RBD image to a block device."""
    print(f"Mapping RBD image: {pool_name}/{image_name}")
    success, output = execute_command(f"sudo rbd map {pool_name}/{image_name}")
    if success:
        return output  # Returns the device path
    return None

def format_device(device_path):
    """Formats the mapped device."""
    print(f"Formatting device: {device_path}")
    return execute_command(f"sudo mkfs.ext4 {device_path}")

def mount_device(device_path, mount_point):
    """Mounts the device at a given mount point."""
    if not os.path.exists(mount_point):
        os.makedirs(mount_point)
    print(f"Mounting device: {device_path} at {mount_point}")
    return execute_command(f"sudo mount {device_path} {mount_point}")

def unmount_device(mount_point):
    """Unmounts the device from the mount point."""
    logging.info(f"Unmounting device from {mount_point}")
    return execute_command(f"sudo umount {mount_point}")

def unmap_rbd_image(pool_name, image_name):
    """Unmaps the RBD image."""
    logging.info(f"Unmapping RBD image: {pool_name}/{image_name}")
    return execute_command(f"sudo rbd unmap {pool_name}/{image_name}")

def read_data(mount_point, file_name):
    """Reads data from a file in the mounted device."""
    try:
        with open(os.path.join(mount_point, file_name), 'r') as file:
            return file.read()
    except IOError as e:
        logging.error(f"Error reading file: {e}")
        return None

def write_data(mount_point, file_name, data):
    """Writes data to a file in the mounted device."""
    try:
        with open(os.path.join(mount_point, file_name), 'w') as file:
            file.write(data)
            return True
    except IOError as e:
        logging.error(f"Error writing file: {e}")
        return False

# Sample usage
pool_name = "myrbdpool"
image_name = "myrbdimage"
image_size = "50M"
total_steps = 5  # Total number of steps in the process

# Create RBD pool and image
if create_rbd_pool(pool_name, 1, total_steps):
    if create_rbd_image(pool_name, image_name, image_size, 2, total_steps):
        # Map, format, and mount the RBD image
        device_path = map_rbd_image(pool_name, image_name)
        if device_path:
            if format_device(device_path):
                mount_point = "/mnt/myrbd"
                if mount_device(device_path, mount_point):
                    # Reading and writing data
                    file_path = "test.txt"
                    test_data = "Hello, RBD World!"
                    write_success = write_data(mount_point, file_path, test_data)
                    if write_success:
                        logging.info("Data written successfully")

                    read_data = read_data(mount_point, file_path)
                    if read_data:
                        logging.info(f"Read data: {read_data}")
