import requests
import pandas as pd

def fetch_features(feature_store_url, feature_list):
    """
    Fetch specified features from the feature store.

    Args:
        feature_store_url (str): URL of the feature store service.
        feature_list (list): List of feature names to retrieve.

    Returns:
        pd.DataFrame: A DataFrame containing the requested features.
    """
    response = requests.get(f"{feature_store_url}/features", params={"features": feature_list})
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    else:
        raise Exception(f"Failed to fetch features: {response.text}")

def update_feature_store(feature_store_url, new_features_df):
    """
    Update the feature store with new features.

    Args:
        feature_store_url (str): URL of the feature store service.
        new_features_df (pd.DataFrame): DataFrame containing new features to update.

    Returns:
        bool: True if update was successful, False otherwise.
    """
    response = requests.post(f"{feature_store_url}/update", json=new_features_df.to_dict(orient='records'))
    return response.status_code == 200

def main():
    feature_store_url = 'http://feature-store-service'  # Replace with actual feature store URL

    # Example feature names to fetch
    feature_list = ['feature1', 'feature2', 'feature3']
    
    # Fetch features
    features_df = fetch_features(feature_store_url, feature_list)
    print("Fetched features:", features_df.head())

    # Example of new features to update in the feature store
    new_features = {
        'feature1': [10, 20],
        'feature2': [30, 40],
        'feature3': [50, 60]
    }
    new_features_df = pd.DataFrame(new_features)

    # Update feature store
    if update_feature_store(feature_store_url, new_features_df):
        print("Feature store updated successfully.")
    else:
        print("Failed to update feature store.")

if __name__ == "__main__":
    main()
