# File: evaluate_model.py

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_model(model, eval_loader):
    """
    Evaluates the performance of a trained model on evaluation data.

    Args:
        model (torch.nn.Module): Trained model.
        eval_loader (DataLoader): DataLoader for evaluation data.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model.eval()  # Set the model to evaluation mode

    # Initialize lists to store true and predicted labels
    true_labels = []
    predictions = []

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        for batch in eval_loader:
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)

            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.extend(batch_predictions)
            true_labels.extend(label_ids)

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

    evaluation_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return evaluation_metrics
