import subprocess
import json
import logging

logging.basicConfig(level=logging.INFO)

def run_ceph_command(command):
    """Runs a Ceph command and returns the output as JSON."""
    try:
        result = subprocess.run(f'sudo {command} -f json', shell=True, check=True, stdout=subprocess.PIPE, text=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing command: {e}")
        return None

def check_stray_daemons(ceph_status):
    """Check for stray daemons."""
    if ceph_status.get('health', {}).get('checks', {}).get('DAEMON_STATUS', {}).get('severity') == 'HEALTH_WARN':
        logging.warning("There are stray daemons not managed by cephadm.")

def check_osd_count(ceph_status):
    """Check if OSD count is less than the default pool size."""
    osd_info = ceph_status.get('osdmap', {}).get('osdmap', {})
    if osd_info.get('num_osds', 0) < osd_info.get('osd_pool_default_size', 3):
        logging.warning("OSD count is less than the default pool size. Consider adding more OSDs.")

def check_pool_application_enabled(ceph_status):
    """Check for pools without an application enabled."""
    pools = ceph_status.get('pools', [])
    for pool in pools:
        if 'application' not in pool:
            logging.warning(f"Pool '{pool.get('pool_name')}' does not have an application enabled.")

def main():
    ceph_status = run_ceph_command('ceph -s')
    if ceph_status:
        check_stray_daemons(ceph_status)
        check_osd_count(ceph_status)
        check_pool_application_enabled(ceph_status)
    else:
        logging.error("Failed to retrieve Ceph status.")

if __name__ == "__main__":
    main()
