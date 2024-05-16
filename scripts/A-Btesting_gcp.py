import random
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Replace these URLs with your actual model endpoint URLs
MODEL_A_URL = "https://us-central1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/YOUR_MODEL_A_ENDPOINT:predict"
MODEL_B_URL = "https://us-central1-aiplatform.googleapis.com/v1/projects/YOUR_PROJECT_ID/locations/us-central1/endpoints/YOUR_MODEL_B_ENDPOINT:predict"

# Define the headers for the request
HEADERS = {
    "Authorization": f"Bearer YOUR_ACCESS_TOKEN",  # Replace with your actual access token
    "Content-Type": "application/json"
}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Randomly choose between Model A and Model B
    model_url = random.choice([MODEL_A_URL, MODEL_B_URL])
    
    try:
        response = requests.post(model_url, headers=HEADERS, json=data)
        response.raise_for_status()
        prediction = response.json()
        return jsonify(prediction)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
