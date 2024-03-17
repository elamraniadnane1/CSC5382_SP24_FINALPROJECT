# File: prepare_data.py
import pandas as pd
import tensorflow_data_validation as tfdv
import re
from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import logging

def prepare_data(tokenized_data, batch_size, test_size=0.2, val_size=0.1):
    """
    Prepares the tokenized data for training. This includes creating TensorDatasets
    and splitting them into training, validation, and test sets.

    Args:
        tokenized_data (dict): A dictionary containing tokenized data.
        batch_size (int): Batch size for the DataLoader.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the training dataset to include in the validation split.

    Returns:
        dict: A dictionary containing DataLoaders for the training, validation, and test sets.
    """
    try:
        # Extract inputs and labels
        input_ids = tokenized_data['input_ids']
        attention_mask = tokenized_data['attention_mask']
        labels = torch.tensor(tokenized_data['labels']).long()

        # Create TensorDataset
        dataset = TensorDataset(input_ids, attention_mask, labels)

        # Split the dataset into train, validation, and test sets
        total_size = len(dataset)
        test_size = int(test_size * total_size)
        train_size = total_size - test_size
        val_size = int(val_size * train_size)
        train_size -= val_size

        train_dataset, test_dataset = random_split(dataset, [train_size + val_size, test_size])
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Prepare output
        prepared_data = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader
        }

        return prepared_data

    except Exception as e:
        logging.error(f"An error occurred during data preparation: {e}")
        raise

def clean_text(text):
    """
    Cleans the text by removing URLs, mentions, hashtags, and extra spaces.

    Args:
        text (str): The text to clean.

    Returns:
        str: Cleaned text.
    """
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)     # Remove mentions
    text = re.sub(r'#', '', text)        # Remove hashtags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces, tabs, and newlines
    return text

def load_and_clean_data(csv_file_path: str) -> pd.DataFrame:
    """
    Loads and cleans the dataset.

    Args:
        csv_file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = pd.read_csv(csv_file_path)
    df['text'] = df['text'].apply(clean_text)
    return df

def validate_data(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Tuple:
    """
    Validates the training and evaluation datasets using TensorFlow Data Validation (TFDV).

    Args:
        train_df (pd.DataFrame): Training dataset.
        eval_df (pd.DataFrame): Evaluation dataset.

    Returns:
        Tuple: Contains the anomalies in training and evaluation datasets.
    """
    # Generate statistics
    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

    # Infer schema
    schema = tfdv.infer_schema(train_stats)

    # Validate statistics
    train_anomalies = tfdv.validate_statistics(statistics=train_stats, schema=schema)
    eval_anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)

    return train_anomalies, eval_anomalies
