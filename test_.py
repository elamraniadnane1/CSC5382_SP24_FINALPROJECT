import os
import mlflow
import mlflow.pytorch
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Define the path to the model files
MODEL_PATH = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'

# Environment setup
# Normally, you'd run these outside of the script to set up your environment
# !pip install mlflow
# !pip install torch
# !pip install transformers

# Load the model
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

# Define the inference function
def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

# Start an MLflow run and log the model
with mlflow.start_run() as run:
    # Log model artifacts
    mlflow.pytorch.log_model(model, "model", registered_model_name="bert-election2020-twitter-stance-biden")
    
    # Save the tokenizer and additional model files as artifacts
    mlflow.log_artifacts(MODEL_PATH, artifact_path="model")

# Serve the model using MLflow's serving tools
# In a real-world scenario, you would run this command in your terminal or deployment environment
# os.system(f"mlflow models serve -m runs:/{run.info.run_id}/model")

print(f"Model and tokenizer have been logged with run id: {run.info.run_id}")
print("To serve the model, run the following command:")
print(f"mlflow models serve -m runs:/{run.info.run_id}/model --port 1234")

# This is how you would serve the model, but you should execute this in your command line, not in the script.
# The --port flag is optional and allows you to specify a custom port for the server.
