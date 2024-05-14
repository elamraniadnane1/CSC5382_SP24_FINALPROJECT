import requests

API_URL = "https://api-inference.huggingface.co/models/elamraniadnane1/bert-election2024-twitter-stance-biden"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "I like you Trump",
})