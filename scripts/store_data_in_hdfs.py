import pandas as pd

def store_data_in_hdfs(data: pd.DataFrame, hdfs_path: str) -> str:
    """
    Store DataFrame data to HDFS.

    Args:
        data (pd.DataFrame): DataFrame to store.
        hdfs_path (str): Path in HDFS to store the data.

    Returns:
        str: Path where the data is stored in HDFS.
    """
    # Your implementation to store data in HDFS goes here
    # Example:
    # data.to_hdf(hdfs_path, key='data', mode='w')
    return hdfs_path  # Returning the same path for simplicity
