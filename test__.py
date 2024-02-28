import mlflow
import mlflow.pytorch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define the path to the model files and the CSV file
#MODEL_PATH = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'
CSV_FILE_PATH = '/home/dino/Desktop/SP24/biden_stance_train_public.csv'

# Select model path here
pretrained_LM_path = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'

# Load tokenizer and model from pretrained path
tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)

# Define the mapping from IDs to labels
id2label = {
    0: "AGAINST",
    1: "FAVOR",
    2: "NONE"
}

# Create a reverse mapping from label names to integers
label2id = {v: k for k, v in id2label.items()}

# Load the dataset from the CSV file
df = pd.read_csv('/home/dino/Desktop/SP24/biden_stance_train_public.csv')
texts = df['text'].tolist()

labels = df['label'].map(label2id).tolist() # Replace 'label_column_name' with the actual name of the label column

# Print the column names to find out what the text column is called
print(df.columns)

# Tokenize the dataset
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Convert labels to a tensor
labels = torch.tensor(labels)
labels = labels.long()

# Create a dataset
dataset = TensorDataset(input_ids, attention_mask, labels)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Start an MLflow run
with mlflow.start_run() as run:
    # Train the model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    for epoch in range(4):  # loop over the dataset multiple times
        total_train_loss = 0
        for step, batch in enumerate(train_loader):
            b_input_ids, b_input_mask, b_labels = batch

            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits

            # Debugging lines
            print(f"Logits shape: {logits.shape}, Labels shape: {b_labels.shape}")
            print(f"Logits dtype: {logits.dtype}, Labels dtype: {b_labels.dtype}")

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), b_labels.view(-1))

            if loss is None:
                print("Loss computation failed.")
                continue

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_loader)            
        print(f"Epoch {epoch+1}, Average Training loss: {avg_train_loss}")
        mlflow.log_metric(f"avg_train_loss_epoch_{epoch+1}", avg_train_loss)
    
    # Evaluate the model
    model.eval()  # Set the model to evaluation mode
    predictions, true_labels = [], []
    
    for batch in val_loader:
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        logits = outputs.logits
        label_ids = b_labels.to('cpu').numpy()
        batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        predictions.extend(batch_predictions)
        true_labels.extend(label_ids)
    
    # Calculate the accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Validation Accuracy: {accuracy}")
    mlflow.log_metric("validation_accuracy", accuracy)

    # Log the model
    mlflow.pytorch.log_model(model, "model")

print(f"Model has been logged with run id: {run.info.run_id}")
