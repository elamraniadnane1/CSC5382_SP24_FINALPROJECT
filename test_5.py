import mlflow
import mlflow.pytorch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np

# Define the path to the model files and the CSV file
CSV_FILE_PATH = '/home/dino/Desktop/SP24/biden_stance_train_public.csv'
pretrained_LM_path = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'

# Set hyperparameters
hyperparams = {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 4,
    "weight_decay": 0.01
}

# Start MLflow run
with mlflow.start_run() as run:
    # Log hyperparameters
    mlflow.log_params(hyperparams)
    
    # Load tokenizer and model from pretrained path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)
    
    # Log model configuration
    mlflow.log_dict(model.config.to_dict(), "model_config.json")

    # Define the mapping from IDs to labels
    id2label = {0: "AGAINST", 1: "FAVOR", 2: "NONE"}
    label2id = {v: k for k, v in id2label.items()}
    
    # Process the CSV
    df = pd.read_csv(CSV_FILE_PATH)
    texts = df['text'].tolist()
    labels = df['label'].map(label2id).tolist()

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = torch.tensor(labels).long()

    # Create a dataset
    dataset = TensorDataset(input_ids, attention_mask, labels)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"])

    # Prepare optimizer
    optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])

    # Training loop
    for epoch in range(hyperparams["epochs"]):
        model.train()
        total_train_loss = 0
        for step, batch in enumerate(train_loader):
            b_input_ids, b_input_mask, b_labels = batch

            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss

            total_train_loss += loss.item()
            loss.backward()

            # Log gradient norms
            grad_norm = np.sqrt(sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None))
            mlflow.log_metric(f"grad_norm_epoch_{epoch+1}", grad_norm)

            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        mlflow.log_metric(f"avg_train_loss_epoch_{epoch+1}", avg_train_loss)

        # Evaluate the model
        model.eval()
        predictions, true_labels = [], []

        for batch in val_loader:
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask)

            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.extend(batch_predictions)
            true_labels.extend(label_ids)

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Log additional metrics with correct formatting
        mlflow.log_metrics({
            f"validation_accuracy_epoch_{epoch+1}": accuracy,
            f"precision_epoch_{epoch+1}": precision,
            f"recall_epoch_{epoch+1}": recall,
            f"f1_score_epoch_{epoch+1}": f1
        })


    # Log dataset and model
    mlflow.log_artifact(CSV_FILE_PATH, "dataset")
    mlflow.pytorch.log_model(model, "model")

    print(f"Model has been logged with run id: {run.info.run_id}")
