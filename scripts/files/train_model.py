# File: train_model.py
import torch
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForSequenceClassification
import mlflow
import numpy as np

def train_model(train_loader, val_loader, model_path, hyperparams):
    """
    Trains a BERT model for sequence classification.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        model_path (str): Path to the pretrained model.
        hyperparams (dict): Hyperparameters for training.

    Returns:
        A trained PyTorch model.
    """
    # Load the pretrained BERT model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])

    # Training loop
    for epoch in range(hyperparams['epochs']):
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_loader):
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()

            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step
        model.eval()
        val_accuracy = 0
        for batch in val_loader:
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)

            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).flatten()
            val_accuracy += (predictions == b_labels).cpu().numpy().mean()

        val_accuracy /= len(val_loader)

        # Log metrics
        mlflow.log_metric(f"train_loss_epoch_{epoch}", avg_train_loss)
        mlflow.log_metric(f"val_accuracy_epoch_{epoch}", val_accuracy)

    return model
