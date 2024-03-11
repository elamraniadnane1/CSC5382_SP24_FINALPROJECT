# Standard Library Imports
import os
import sys
import logging
from typing import Dict, Annotated
import pandas as pd
import re
# Third-party Library Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import mlflow
import mlflow.pytorch
import tensorflow_data_validation as tfdv
import subprocess

# ZenML Imports
from zenml.pipelines import pipeline
from zenml.steps import step, Output
import pandas as pd

# Add script directory to path for module imports
sys.path.append('/home/dino/Desktop/SP24/')
from scripts import (
    prepare_data, train_model,
    evaluate_model, track_experiment
)
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AdamW

# Constants and Hyperparameters
CSV_FILE_PATH = '/home/dino/Desktop/SP24/scripts/biden_stance_public.csv'
PRETRAINED_LM_PATH = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'
HYPERPARAMS = {
    "batch_size": 8,
    "learning_rate": 2e-5,
    "epochs": 2,
    "weight_decay": 0.01
}
print("Hyperparameters:", HYPERPARAMS)

DATA_FILE_PATH = '/home/dino/Desktop/SP24/scripts/processed_data.csv'
# Define ZenML steps
# 1. Data Loading Step
@step
def load_data() -> Output(data=pd.DataFrame):
    data = pd.read_csv(CSV_FILE_PATH)
    return data

# 2. Data Preprocessing Step
@step
def preprocess_data(data: pd.DataFrame) -> Output(train_set=pd.DataFrame, test_set=pd.DataFrame):
    # Load the evaluation dataset
    EVAL_CSV_FILE_PATH = '/home/dino/Desktop/SP24/scripts/biden_stance_public_3_labeled.csv'
    eval_df = pd.read_csv(EVAL_CSV_FILE_PATH)

    # Generate and visualize statistics for both datasets using TensorFlow Data Validation (TFDV)
    train_stats = tfdv.generate_statistics_from_dataframe(data)
    eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

    # Visualize statistics
    print("Training Dataset Statistics:")
    tfdv.visualize_statistics(train_stats)
    print("Evaluation Dataset Statistics:")
    tfdv.visualize_statistics(eval_stats)

    # Infer schema from training data
    schema = tfdv.infer_schema(train_stats)

    # Validate statistics against the schema
    train_anomalies = tfdv.validate_statistics(statistics=train_stats, schema=schema)
    eval_anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)

    # Display any anomalies
    print("Anomalies in Training Dataset:")
    tfdv.display_anomalies(train_anomalies)
    print("Anomalies in Evaluation Dataset:")
    tfdv.display_anomalies(eval_anomalies)

    # Function to clean text data
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'@\S+', '', text)  # Remove mentions
        text = re.sub(r'#', '', text)  # Remove hashtags
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces, tabs, and newlines
        return text

    # Clean the 'text' column in both datasets
    data['text'] = data['text'].apply(clean_text)
    eval_df['text'] = eval_df['text'].apply(clean_text)

    # Save the cleaned datasets
    data.to_csv(CSV_FILE_PATH, index=False)
    eval_df.to_csv(EVAL_CSV_FILE_PATH, index=False)

    # Save the schema
    SCHEMA_FILE = '/home/dino/Desktop/SP24/scripts/schema.txt'
    tfdv.write_schema_text(schema, SCHEMA_FILE)

    # Split the training dataset into train and test sets
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

    return train_set, test_set


# You need to implement this according to your label encoding
def label_to_number(label):
    # Example encoding: {'label1': 0, 'label2': 1, ...}
    label_encoding = {'your_label_1': 0, 'your_label_2': 1}  # Update with your labels
    return label_encoding.get(label, -1)  # Return -1 or any default value for unknown labels

# 3. Model Training Step
@step
def train(train_set: pd.DataFrame, test_set: pd.DataFrame) -> ClassifierMixin:
    # Load the tokenizer for the BERT model
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_LM_PATH)

    # Function to tokenize and prepare the dataset
    def tokenize_and_prepare(dataset):
        input_ids = []
        attention_masks = []
        labels = []

        for _, row in dataset.iterrows():
            encoded_dict = tokenizer.encode_plus(
                row['text'],  # Text to encode
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=64,  # Pad & truncate all sentences
                truncation=True,  # Explicitly truncate examples to max length
                padding='max_length',  # Pad to max_length
                return_attention_mask=True,  # Construct attention masks
                return_tensors='pt',  # Return pytorch tensors
            )
            
            input_ids.append(encoded_dict['input_ids'][0])
            attention_masks.append(encoded_dict['attention_mask'][0])
            
            # Convert labels to numeric and add to the list
            label = row['label']
            numeric_label = label_to_number(label)
            labels.append(numeric_label)

        # Convert lists to tensors
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.tensor(labels, dtype=torch.long)

        return TensorDataset(input_ids, attention_masks, labels)

    # Tokenize and prepare datasets
    train_dataset = tokenize_and_prepare(train_set)
    val_dataset = tokenize_and_prepare(test_set)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),  # Random sampling for training
        batch_size=HYPERPARAMS['batch_size']
    )

    val_loader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),  # Sequential sampling for validation
        batch_size=HYPERPARAMS['batch_size']
    )
    # Tokenization, DataLoader creation, etc. should be here.
    # For this example, I'm assuming that train_loader and val_loader are already created.


    # Load the pretrained BERT model
    model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_LM_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=HYPERPARAMS['learning_rate'], weight_decay=HYPERPARAMS['weight_decay'])

    # Training loop
    for epoch in range(HYPERPARAMS['epochs']):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            # Unpack the training batch from the dataloader
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)

            # Clear previously calculated gradients
            optimizer.zero_grad()

            # Perform a forward pass and calculate the training loss
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients
            loss.backward()

            # Update parameters and take a step using the computed gradient
            optimizer.step()

        # Calculate the average loss over the training data
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation step
        model.eval()
        val_accuracy = 0
        for batch in val_loader:
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_input_mask = b_input_mask.to(device)
            b_labels = b_labels.to(device)

            # Evaluate the model on this batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask)

            # Move logits and labels to CPU for further calculation
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).flatten()
            val_accuracy += (predictions == b_labels).cpu().numpy().mean()

        # Calculate the average accuracy over the validation data
        val_accuracy /= len(val_loader)


        return model

# 4. Model Evaluation Step
@step
def evaluate(model: ClassifierMixin, test_set: pd.DataFrame) -> Dict:
    # Assuming the 'evaluate_model' function evaluates the model and returns metrics
    metrics = evaluate_model(model, test_set)
    return metrics

# 5. Model Tracking Step
@step
def track(model: ClassifierMixin, metrics: Dict):
    # Assuming the 'track_experiment' function tracks the experiment in MLflow
    track_experiment(model, metrics)


# 7. Pipeline Definition
@pipeline
def stance_detection_pipeline(
    load_data_step,
    preprocess_data_step,
    train_step,
    evaluate_step,
    track_step
):
    data = load_data_step()
    train_set, test_set = preprocess_data_step(data)
    model = train_step(train_set, test_set)
    metrics = evaluate_step(model, test_set)
    track_step(model, metrics)
    
stance_pipeline = stance_detection_pipeline(
    load_data_step=load_data(),
    preprocess_data_step=preprocess_data(),
    train_step=train(),
    evaluate_step=evaluate(),
    track_step=track(),
)
stance_pipeline.run()


