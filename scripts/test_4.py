import mlflow
import mlflow.pytorch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Define the path to the model files and the CSV file
CSV_FILE_PATH = '/home/dino/Desktop/SP24/biden_stance_train_public.csv'
pretrained_LM_path = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'

# Set hyperparameters
hyperparams = {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 4
}

# Start MLflow run
with mlflow.start_run() as run:
    # Log hyperparameters
    mlflow.log_params(hyperparams)
    
    # Load tokenizer and model from pretrained path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)
    
    # Define the mapping from IDs to labels
    id2label = {0: "AGAINST", 1: "FAVOR", 2: "NONE"}
    label2id = {v: k for k, v in id2label.items()}
    
    # Initialize lists to store tokenized inputs and labels
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    # Process the CSV in chunks
    chunksize = 10000  # Adjust chunksize according to your system's memory
    for chunk in pd.read_csv(CSV_FILE_PATH, chunksize=chunksize):
        texts = chunk['text'].tolist()
        labels = chunk['label'].map(label2id).tolist()

        # Tokenize the dataset
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        input_ids_list.append(inputs["input_ids"])
        attention_mask_list.append(inputs["attention_mask"])
        labels_list.append(torch.tensor(labels).long())

    # Concatenate all chunks
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_mask = torch.cat(attention_mask_list, dim=0)
    labels = torch.cat(labels_list, dim=0)

    # Create a dataset
    dataset = TensorDataset(input_ids, attention_mask, labels)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    batch_size = hyperparams["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["learning_rate"])

    for epoch in range(hyperparams["epochs"]):
        total_train_loss = 0
        for step, batch in enumerate(train_loader):
            b_input_ids, b_input_mask, b_labels = batch

            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 3), b_labels.view(-1))

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        mlflow.log_metric(f"avg_train_loss_epoch_{epoch+1}", avg_train_loss)

    # Evaluate the model
    model.eval()
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

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    
    # Log additional metrics
    mlflow.log_metrics({
        "validation_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    # Log dataset
    mlflow.log_artifact(CSV_FILE_PATH, "dataset")

    # Log the model
    mlflow.pytorch.log_model(model, "model")

    print(f"Model has been logged with run id: {run.info.run_id}")
