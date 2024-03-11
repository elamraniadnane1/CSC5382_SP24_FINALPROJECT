import requests
import logging
import requests
import logging

API_URL = "https://api-inference.huggingface.co/models/kornosk/bert-election2020-twitter-stance-biden"

# Define the label interpretations according to the model's documentation or output.
LABEL_MAP = {
    'LABEL_0': 'Neutral Stance',
    'LABEL_1': 'Pro-Biden Stance',
    'LABEL_2': 'Anti-Biden Stance',
}

def interpret_labels(output):
    # Check if the output is a list of results
    if isinstance(output, list):
        # Iterate through each item in the list
        for item in output:
            # Check if the item is a dictionary and has the 'label' key
            if isinstance(item, dict) and 'label' in item:
                label = item['label']
                score = item['score']
                interpretation = LABEL_MAP.get(label, 'Unknown')
                print(f"Label: {label}, Score: {score:.2f}, Interpretation: {interpretation}")
            # If the item is a list, it's possibly the format [{'label': ..., 'score': ...}]
            elif isinstance(item, list):
                for result in item:
                    if isinstance(result, dict):
                        label = result.get('label')
                        score = result.get('score')
                        interpretation = LABEL_MAP.get(label, 'Unknown')
                        print(f"Label: {label}, Score: {score:.2f}, Interpretation: {interpretation}")
                    else:
                        print("Unexpected format in nested list.")
            else:
                print("Unexpected format in result list.")
    else:
        print("Output format not recognized.")


# Improved function with type hints, error handling, and logging
def query(payload: dict, api_key: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        logging.error(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        logging.error(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as err:
        logging.error(f"An error occurred: {err}")
    return {}

# Example usage:
api_key = "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"  # Replace with your actual API key
output = query({
    "inputs": "I like you !",
}, api_key)

# Print the interpreted labels
interpret_labels(output)

