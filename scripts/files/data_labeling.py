import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
import os

# Define the path to the model files and the CSV file
CSV_FILE_PATH = '/home/dino/Desktop/SP24/scripts/biden_new_2024_2.csv'
pretrained_LM_path = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)
model.eval()  # Set the model to evaluation mode

# Load the dataset
df = pd.read_csv(CSV_FILE_PATH)

# Prepare the data for the model
texts = df['text'].tolist()
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
dataset = TensorDataset(input_ids, attention_mask)

# DataLoader for handling the dataset
loader = DataLoader(dataset, batch_size=16)

# Function to get predictions from the model
def get_predictions(model, data_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask = batch
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).tolist()
            predictions.extend(preds)

    return predictions

# Get predictions
predicted_labels = get_predictions(model, loader)

# Mapping model output to actual labels
id2label = {0: "AGAINST", 1: "FAVOR", 2: "NEUTRAL"}
df['label'] = [id2label[label] for label in predicted_labels]

# Save the labeled dataset
output_csv_path = os.path.join(os.path.dirname(CSV_FILE_PATH), 'labeled_dataset.csv')
df.to_csv(output_csv_path, index=False)

print(f"Dataset labeling completed and saved as '{output_csv_path}'")
