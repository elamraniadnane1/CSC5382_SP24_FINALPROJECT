from zenml import pipeline, step
import mlflow
import mlflow.pytorch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim import AdamW
import tensorflow_data_validation as tfdv
import subprocess
from zenml.pipelines import pipeline
from zenml.steps import step, Output
import logging
from typing import Tuple, Annotated
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.base import ClassifierMixin
from sklearn.datasets import load_iris

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
HDFS_PATH = ''
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

# Enable Git LFS for large files
print("Enabling Git LFS for large files...")
run_command("git lfs install")
# Add large files to track with Git LFS
run_command("git lfs track '*.pt'")  # Tracking PyTorch model files
run_command("git lfs track '*.csv'")  # Tracking large CSV datasets

# Add .gitattributes to the staging area
run_command("git add .gitattributes")
# Usage
script_folder_path = '/home/dino/Desktop/SP24/scripts'
check_and_create_script_files(script_folder_path)
# Importing necessary functions from scripts
from scripts.load_data import load_data
from scripts.store_data_in_hdfs import store_data_in_hdfs
from scripts.data_validation import validate_data
from scripts.tokenize_data import tokenize
from scripts.prepare_data import prepare_data
from scripts.train_model import train_model
from scripts.evaluate_model import evaluate_model
from scripts.save_model import save_model
from scripts.experiment_tracking import track_experiment
from scripts.model_interpretability import interpret_model
from scripts.bias_mitigation import mitigate_bias
from scripts.documentation import document
from scripts.feature_store_implementation import implement_feature_store

# Import additional scripts as needed
from scripts.data_partitioning import partition_data
from scripts.feedback_loop import implement_feedback_loop
from scripts.model_deployment import deploy_model
from scripts.model_comparison import compare_models
from scripts.error_analysis import analyze_errors
from scripts.model_registry import register_model
from scripts.feature_engineering import feature_engineering
from scripts.hyperparameter_tuning import tune_hyperparameters
from scripts.feature_selection import select_features
from scripts.emoji_handling import handle_emojis
from scripts.text_normalization import normalize_text
from scripts.schema_validation import validate_schema
# ... continue importing other necessary scripts
# Continuing the import statements
from scripts.feedback_loop import run_feedback_loop
from scripts.data_preprocessing import preprocess_data
from scripts.feature_engineering import perform_feature_engineering
from scripts.hyperparameter_tuning import tune_hyperparameters
from scripts.model_deployment import deploy_model
from scripts.model_packaging import package_model
from scripts.model_test import test_model
from scripts.anomaly_detection_correction import correct_anomalies
from scripts.data_statistics_generation import generate_data_statistics
from scripts.ETL_pipeline_storedata import etl_store_data
from scripts.evaluation_set_anomaly_checker import check_evaluation_set_anomalies
from scripts.feature_selection import select_features
from scripts.one_hot_encoding import apply_one_hot_encoding
from scripts.ordinal_feature_transformation import transform_ordinal_features
from scripts.schema_inference_validator import validate_schema_inference
from scripts.schema_revision_tool import revise_schema
from scripts.sentiment_analysis import analyze_sentiment
from scripts.stop_words_removal import remove_stop_words
from scripts.text_normalization import normalize_text
from scripts.token_frequency import calculate_token_frequency
from scripts.tokenize_data import tokenize_data
from scripts.bag_of_words import create_bag_of_words
from scripts.bag_of_ngrams import create_bag_of_ngrams
from scripts.categorical_to_numerical import convert_categorical_to_numerical
from scripts.custom_feature_extraction import extract_custom_features
from scripts.data_labeling import label_data
from scripts.data_partitionning import partition_data
from scripts.data_schema_inference import infer_data_schema
from scripts.emoji_handling import handle_emojis
from scripts.imbalanced_data_handling import handle_imbalanced_data
from scripts.json_data_handler import handle_json_data
from scripts.lemmatization import apply_lemmatization
from scripts.merge_csv import merge_csv_files
from scripts.tfidf_vectorization import apply_tfidf_vectorization
from scripts.word_embeddings_generation import generate_word_embeddings
# Continue importing additional scripts as needed
# Ensure each of the imported functions exists and works as intended in their respective script files
# Ensure you have created these additional `.py` files with the respective functions 
# to be used in the pipeline. Each function should be designed to fulfill a specific task 
# in the pipeline and should be modular for easy integration.

import os

def check_and_create_script_files(script_folder_path):
    required_files = [
        "anomaly_detection_correction.py", "data_statistics_generator.py", "sentiment_analysis.py",
        "bag_of_ngrams.py", "data_validation.py", "setup.py",
        "bag_of_words.py", "documentation.py", "split_dataset.py",
        "baseline_model_setup.py", "emoji_handling.py", "stop_words_removal.py",
        "bias_mitigation.py", "ETL_pipeline_storedata.py", "store_data_in_hdfs.py",
        "evaluation_set_anomaly_checker.py", "experiment_tracking.py", "text_normalization.py",
        "feature_engineering_pipeline.py", "feature_engineering.py", "tf_idf_transformation.py",
        "feature_selection.py", "feature_store_implementation.py", "tfidf_vectorization.py",
        "feature_store_integration.py", "feedback_loop.py", "token_frequency.py",
        "format_extract_biden_tweets_csv.py", "one_hot_encoding.py", "tokenization.py",
        "categorical_to_numerical.py", "ordinal_feature_transformation.py", "tokenize_data.py",
        "convert_notebook_toscript.py", "prepare_data.py", "train_model.py",
        "custom_feature_extraction.py", "preprocessing_pipeline_builder.py", "tweets_scrapping_api.py",
        "data_labeling.py", "ignore_large_files.py", "unzip_folder.py",
        "data_partitionning.py", "imbalanced_data_handling.py", "utils.py",
        "data_pipeline_setup.py", "json_data_handler.py", "word_embeddings_generation.py",
        "data_preprocessing.py", "lemmatization.py", "model_deployment.py",
        "data_schema_inference.py", "load_data.py", "model_evaluation.py",
        "data_statistics_generation.py", "merge_csv.py", "model_interpretability.py",
        "model_packaging.py", "model_test.py", "model_training.py",
        "hyperparameter_tuning.py", "model_test_2.py", "schema_inference_validator.py",
        "schema_revision_tool.py", "schema_validation.py",
        # Additional files from Jupyter Notebooks
        "Milestone2.ipynb", "MILESTONE2.py", "ML_pipeline_constructor.py",
        "ML_pipeline_mlflow.py", "ml_pipeline_zenml.py", "save_model_folder.py",
        "Tw_2024_2.csv", "Tw_2024_3.csv", "Tw_2024.csv",
        "biden_new_2024_2.csv", "biden_new_2024_2_notformat.csv", "biden_new_2024.csv",
        "biden_stance_public_2_labeled.csv", "biden_stance_public_3_labeled.csv",
        "biden_stance_public.csv", "biden_stance_test_public.csv", "biden_stance_train_public.csv",
        "git_utils.py", "schema.txt"
        # Add more file names if necessary
    ]

    for file in required_files:
        file_path = os.path.join(script_folder_path, file)
        if not os.path.exists(file_path):
            print(f"Creating missing file: {file}")
            with open(file_path, 'w') as f:
                f.write("# Placeholder for " + file)

# Usage
script_folder_path = '/home/dino/Desktop/SP24/scripts'
check_and_create_script_files(script_folder_path)


# Define ZenML steps
# Define additional steps similarly...

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
def store_data_in_hdfs_step(data: pd.DataFrame, hdfs_path: str) -> Annotated[Output[str], step("Store Data in HDFS")]:
    """
    ZenML step to store data in HDFS.

    Args:
        data (pd.DataFrame): DataFrame to store.
        hdfs_path (str): Path in HDFS to store the data.

    Returns:
        str: Path where the data is stored in HDFS.
    """
    # Store data using the store_data_in_hdfs function from scripts.store_data_in_hdfs
    if not os.path.exists(hdfs_path):
        raise FileNotFoundError(f"HDFS path not found at {hdfs_path}")
    stored_path = store_data_in_hdfs(data, hdfs_path)
    return stored_path

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
def track_experiment_step() -> Annotated[Output[str], step("Track Experiment")]:
    """
    ZenML step to track an experiment.

    Returns:
        str: Experiment ID.
    """
    # Track experiment using the track_experiment function from scripts.experiment_tracking
    experiment_id = track_experiment()
    return experiment_id



@step
def interpret_model_step(model: object) -> Annotated[Output[str], step("Interpret Model")]:
    """
    ZenML step to interpret a trained machine learning model.

    Args:
        model (object): Trained machine learning model.

    Returns:
        str: Interpretation result.
    """
    # Interpret model using the interpret_model function from scripts.model_interpretability
    interpretation = interpret_model(model)
    return interpretation

@step
def mitigate_bias_step(data: dict) -> Annotated[Output[dict], step("Mitigate Bias")]:
    """
    ZenML step to mitigate bias in data.

    Args:
        data (dict): Data to mitigate bias.

    Returns:
        dict: Mitigated data.
    """
    # Mitigate bias using the mitigate_bias function from scripts.bias_mitigation
    mitigated_data = mitigate_bias(data)
    return mitigated_data

@step
def implement_feature_store_step(data: dict) -> Annotated[Output[str], step("Implement Feature Store")]:
    """
    ZenML step to implement a feature store.

    Args:
        data (dict): Data to be used to implement the feature store.

    Returns:
        str: Path to the implemented feature store.
    """
    # Implement feature store using the implement_feature_store function from scripts.feature_store_implementation
    feature_store_path = implement_feature_store(data)
    return feature_store_path


# Assemble ZenML pipeline
@pipeline


# Run the pipeline
