import subprocess
import logging

logging.basicConfig(level=logging.INFO)

def run_command(command):
    """Run a shell command and return its output."""
    try:
        subprocess.run(command, shell=True, check=True)
        logging.info(f"Command executed successfully: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing command: {e}")

def delete_rbd_image(image_name, pool_name):
    """Delete a specific RBD image."""
    logging.info(f"Deleting RBD image '{image_name}' from pool '{pool_name}'")
    run_command(f"sudo rbd rm {pool_name}/{image_name}")

def delete_ceph_pool(pool_name):
    """Delete a specific Ceph pool."""
    logging.info(f"Deleting Ceph pool '{pool_name}'")
    run_command(f"sudo ceph osd pool delete {pool_name} {pool_name} --yes-i-really-really-mean-it")

def main():
    # Replace 'myrbdimage' and 'myrbdpool' with your actual RBD image and pool names
    rbd_image_name = "myrbdimage"
    rbd_pool_name = "myrbdpool"

    # Delete the RBD image
    delete_rbd_image(rbd_image_name, rbd_pool_name)

    # Delete the Ceph pool
    delete_ceph_pool(rbd_pool_name)

if __name__ == "__main__":
    main()
