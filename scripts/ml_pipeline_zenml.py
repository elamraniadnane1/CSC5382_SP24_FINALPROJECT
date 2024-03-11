# Standard Library Imports
import os
import sys
from typing import Dict, Tuple, Annotated

# Third-party Library Imports
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tensorflow_data_validation as tfdv
import subprocess

# ZenML Imports
from zenml.pipelines import pipeline
from zenml.steps import step, Output
import logging

# Custom Script Imports
from scripts.load_data import load_data
from scripts.tokenize_data import tokenize_data
from scripts.prepare_data import prepare_data
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model
from scripts.save_model import save_model
from scripts.experiment_tracking import track_experiment
from scripts.feature_store_implementation import implement_feature_store


# Define constants like CSV_FILE_PATH and hyperparameters here
# Set hyperparameters
hyperparams = {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 4,
    "weight_decay": 0.01
}

print("Hyperparameters:", hyperparams)
# Define the path to the model files and the CSV file
CSV_FILE_PATH = '/home/dino/Desktop/SP24/scripts/biden_stance_public_3_labeled.csv'
pretrained_LM_path = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'


# Define function to run shell commands
def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        # Enhanced error logging
        print(f"Error occurred during command '{command}': {stderr.decode().strip()}")
    return stdout.decode().strip()
# Initialize Git repository
print("Initializing Git repository...")
run_command("git init")


# Define ZenML steps

@step
def load_data_step() -> Annotated[Output[Dict], step("Load Data")]:
    """
    ZenML step to load data.

    Returns:
        Dict: Loaded data.
    """
    CSV_FILE_PATH = '/home/dino/Desktop/SP24/scripts/biden_stance_public_3_labeled.csv'

    logging.info(f"Attempting to load data from {CSV_FILE_PATH}")
    
    # Check if the file exists
    if not os.path.exists(CSV_FILE_PATH):
        logging.error(f"CSV file not found at {CSV_FILE_PATH}")
        raise FileNotFoundError(f"CSV file not found at {CSV_FILE_PATH}")

    # Load data using the load_data function from scripts.load_data
    try:
        data = load_data(CSV_FILE_PATH)
        logging.info("Data successfully loaded.")
        # Optional: Add any additional data checks or summaries here
        # For example, checking the number of rows and columns
        logging.info(f"Data shape: {len(data)}, {len(data[next(iter(data))])} (rows, columns)")
        return data
    except Exception as e:
        logging.error(f"An error occurred while loading the data: {e}")
        raise



@step
def tokenize_data_step(data: dict) -> Annotated[Output[dict], step("Tokenize Data")]:
    """
    ZenML step to tokenize data.

    Args:
        data (dict): Data to tokenize.

    Returns:
        dict: Tokenized data.
    """
    # Define the path to the tokenizer
    tokenizer_path = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'

    logging.info("Starting data tokenization.")
    # Tokenize data using the tokenize_data function from scripts.tokenize_data
    tokenized_data = tokenize_data(data, tokenizer_path)
    return tokenized_data


@step
def prepare_data_step(csv_file_path: str, eval_csv_file_path: str, batch_size: int) -> Output(train_loader=DataLoader, val_loader=DataLoader, test_loader=DataLoader, train_anomalies=dict, eval_anomalies=dict):
    """
    ZenML step to load, clean, validate, and prepare data.

    Args:
        csv_file_path (str): Path to the training CSV file.
        eval_csv_file_path (str): Path to the evaluation CSV file.
        batch_size (int): Batch size for the DataLoader.

    Returns:
        DataLoader: DataLoaders for the training, validation, and test sets.
        dict: Anomalies found in the training and evaluation datasets.
    """
    # Load and clean the training and evaluation data
    train_df = load_and_clean_data(csv_file_path)
    eval_df = load_and_clean_data(eval_csv_file_path)

    # Validate the data
    train_anomalies, eval_anomalies = validate_data(train_df, eval_df)

    # Tokenize and prepare the data for model training
    # Assuming tokenization is needed, otherwise, directly prepare the data
    # tokenized_data = some_tokenization_function(train_df)
    prepared_data = prepare_data(tokenized_data, batch_size)

    return prepared_data['train_loader'], prepared_data['val_loader'], prepared_data['test_loader'], train_anomalies, eval_anomalies

@step
def train_model_step(train_loader: DataLoader, val_loader: DataLoader) -> Annotated[Output[object], step("Train Model")]:
    """
    ZenML step to train a machine learning model.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.

    Returns:
        object: Trained machine learning model.
    """
    model_path = '/path/to/pretrained/model'  # Update this path as necessary
    hyperparams = {
        "batch_size": 16,
        "learning_rate": 2e-5,
        "epochs": 4,
        "weight_decay": 0.01
    }

    # Train the model
    model = train_model(train_loader, val_loader, model_path, hyperparams)

    # Save the trained model
    # e.g., torch.save(model.state_dict(), 'path/to/save/model.pth')

    return model

@step
def evaluate_model_step(model: object, eval_loader: DataLoader) -> Annotated[Output[dict], step("Evaluate Model")]:
    """
    ZenML step to evaluate a machine learning model.

    Args:
        model (object): Trained machine learning model.
        eval_loader (DataLoader): DataLoader for evaluation data.

    Returns:
        dict: Evaluation metrics.
    """
    # Evaluate the model
    evaluation_metrics = evaluate_model(model, eval_loader)

    # Log metrics
    # Optionally, log metrics with MLflow or any other tracking system

    return evaluation_metrics



@step
def save_model_step(model: object, save_dir: str = '/path/to/save/models/') -> Annotated[Output[str], step("Save Model")]:
    """
    ZenML step to save a trained machine learning model.

    Args:
        model (object): Trained machine learning model.
        save_dir (str): Directory to save the model.

    Returns:
        str: Path where the model is saved.
    """
    # Save the model
    model_path = save_model(model, save_dir)

    return model_path



@step
def track_experiment_step(hyperparams: dict, evaluation_metrics: dict, model_path: str, other_artifacts: dict = None) -> Annotated[Output[str], step("Track Experiment")]:
    """
    ZenML step to track an experiment.

    Args:
        hyperparams (dict): Hyperparameters used for training.
        evaluation_metrics (dict): Evaluation metrics of the model.
        model_path (str): Path to the saved model.
        other_artifacts (dict, optional): Additional artifacts to log.

    Returns:
        str: Experiment ID.
    """
    # Track the experiment
    experiment_id = track_experiment(hyperparams, evaluation_metrics, model_path, other_artifacts)
    
    return experiment_id




@step
def interpret_model_step(model: object, test_loader: DataLoader) -> Output(str, step("Interpret Model")):
    """
    ZenML step to interpret a trained machine learning model.

    Args:
        model (object): Trained machine learning model.
        test_loader (DataLoader): DataLoader containing test data for interpretation.

    Returns:
        str: Interpretation result.
    """
    # Call the interpret_model function from scripts.interpret_model
    # Pass the model and test_loader as arguments to the interpret_model function
    interpretation = interpret_model(model, test_loader)

    # Return the interpretation result
    return interpretation


@step
def implement_feature_store_step(data: dict) -> Output(str, step("Implement Feature Store")):
    """
    ZenML step to implement a feature store.

    Args:
        data (dict): Data to be used to implement the feature store.

    Returns:
        str: Path to the implemented feature store.
    """
    feature_store_path = implement_feature_store(data)
    return feature_store_path


# Assemble ZenML pipeline
@pipeline


# Assemble ZenML pipeline
@pipeline
def tweet_analysis_pipeline(
    load_data,
    store_data_in_hdfs,
    tokenize_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    track_experiment,
    interpret_model,
    implement_feature_store
):
    data = load_data()
    hdfs_path = store_data_in_hdfs(data)
    tokenized_data = tokenize_data(data)
    train_loader, val_loader, test_loader, train_anomalies, eval_anomalies = prepare_data(
        csv_file_path=CSV_FILE_PATH,
        eval_csv_file_path=CSV_FILE_PATH,  # Adjust the eval_csv_file_path as necessary
        batch_size=hyperparams['batch_size']
    )
    model = train_model(train_loader, val_loader)
    evaluation_metrics = evaluate_model(model, val_loader)
    model_path = save_model(model)
    experiment_id = track_experiment(hyperparams, evaluation_metrics, model_path)
    interpretation = interpret_model(model, test_loader)
    feature_store_path = implement_feature_store(data)

    # You can use the outputs of each step for further steps or simply pass them along

# Run the pipeline
def run():
    repo = Repository()
    my_pipeline = repo.get_pipeline(pipeline_name='tweet_analysis_pipeline')
    my_pipeline.run()

# Execute the function to run the pipeline
if __name__ == "__main__":
    run()