# Imports (add necessary imports for your tools and libraries)
# General imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_data_validation as tfdv
import mlflow
import mlflow.tensorflow
# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# ZenML imports
from zenml.steps import step
from zenml.pipelines import pipeline
# For Twitter API (if needed for real-time data)
import tweepy

# Other imports you might need (depending on your exact requirements)
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import json
import requests
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import uuid
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


@step
def load_data(csv_file_path):
    """
    Step to load data from a CSV file.

    Args:
    csv_file_path (str): Path to the CSV file.

    Returns:
    pandas.DataFrame: Loaded data.
    """
    # Load the dataset
    data = pd.read_csv(csv_file_path)

    # Return the loaded data
    return data


# Constants and Hyperparameters
CSV_FILE_PATH = 'C:\\Users\\LENOVO\\Desktop\\SP24\\scripts\\dataset.csv'
PRETRAINED_LM_PATH = 'C:\\Users\\LENOVO\\Desktop\\SP24\\bert-election2020-twitter-stance-biden'
HYPERPARAMS = {
    "batch_size": 8,
    "learning_rate": 2e-5,
    "epochs": 2,
    "weight_decay": 0.01,
    'max_grad_norm': 1.0,
    'lr_step_size': 1,
    'lr_gamma': 0.1,
    
}
print("Hyperparameters:", HYPERPARAMS)
import mlflow

mlflow.set_tracking_uri('http://127.0.0.1:5000')  # Replace with your server URI


@step
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Step for data preprocessing which includes cleaning and tokenization.

    Args:
        data (pandas.DataFrame): The loaded data.

    Returns:
        pandas.DataFrame: Preprocessed data.
    """
    def clean_text(text):
        """
        Function to clean text data.

        Args:
            text (str): Text to be cleaned.

        Returns:
            str: Cleaned text.
        """
        # Removing URLs
        text = re.sub(r'http\S+', '', text)
        # Removing special characters and numbers
        text = re.sub(r'[^A-Za-z\s]', '', text)
        # Converting to lowercase
        text = text.lower().strip()
        return text

    if 'text' in data.columns:
        # Efficiently clean the text data
        data['text'] = data['text'].astype(str).apply(clean_text)
    else:
        raise ValueError("The dataframe does not have a 'text' column.")

    # Encode labels if present
    if 'label' in data.columns:
        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])

    return data


@step
def visualize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Step for data visualization including checking for missing values.

    Args:
        data (pandas.DataFrame): The preprocessed data.

    Returns:
        pandas.DataFrame: The data (if the next step in the pipeline requires it).
    """
    try:
        # Generate statistics from the dataset using TFDV
        stats = tfdv.generate_statistics_from_dataframe(data)
        tfdv.visualize_statistics(stats)
    except Exception as e:
        print(f"Error occurred in TFDV statistics generation: {e}")

    try:
        # Visualizing Missing Values
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        missing_df = pd.DataFrame({'Feature': missing_values.index, 'MissingValues': missing_values, 'Percentage': missing_percentage})

        # Filter out features with no missing values
        missing_df = missing_df[missing_df['MissingValues'] > 0].sort_values('Percentage', ascending=False)

        # Plotting missing values (if there are any)
        if not missing_df.empty:
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Percentage', y='Feature', data=missing_df)
            plt.title('Percentage of Missing Values per Feature')
            plt.xlabel('Percentage')
            plt.ylabel('Feature')
            plt.show()
        else:
            print("No missing values found in the dataset.")
    except Exception as e:
        print(f"Error occurred during missing values visualization: {e}")

    return data



@step
def split_data_1(data: pd.DataFrame) -> pd.DataFrame:
    """
    Step to split the data into training, validation, and testing sets, but only return the training set.

    Args:
        data (pandas.DataFrame): The DataFrame containing the preprocessed data.

    Returns:
        pandas.DataFrame: The training data set.
    """
    # Define the size for your test and validation sets
    test_size = 0.2
    validation_size = 0.1

    # Initial split to separate out the test set
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    # Calculate the adjusted validation size based on the remaining data after test split
    adjusted_validation_size = validation_size / (1 - test_size)

    # Split the remaining data into training and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=adjusted_validation_size, random_state=42)

    # Return only the training data
    return train_data


@step
def split_data_2(data: pd.DataFrame) -> pd.DataFrame:
    """
    Step to split the data into training, validation, and testing sets, but only return the validation set.

    Args:
        data (pandas.DataFrame): The DataFrame containing the preprocessed data.

    Returns:
        pandas.DataFrame: The validation data set.
    """
    # Define the size for your test and validation sets
    test_size = 0.2
    validation_size = 0.1

    # Initial split to separate out the test set
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    # Calculate the adjusted validation size based on the remaining data after test split
    adjusted_validation_size = validation_size / (1 - test_size)

    # Split the remaining data into training and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=adjusted_validation_size, random_state=42)

    # Return only the validation data
    return val_data
@step
def split_data_3(data: pd.DataFrame) -> pd.DataFrame:
    """
    Step to split the data into training, validation, and testing sets, but only return the testing set.

    Args:
        data (pandas.DataFrame): The DataFrame containing the preprocessed data.

    Returns:
        pandas.DataFrame: The testing data set.
    """
    # Define the size for your test and validation sets
    test_size = 0.2
    validation_size = 0.1

    # Initial split to separate out the test set
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)

    # Calculate the adjusted validation size based on the remaining data after test split
    adjusted_validation_size = validation_size / (1 - test_size)

    # Split the remaining data into training and validation sets
    train_data, val_data = train_test_split(train_val_data, test_size=adjusted_validation_size, random_state=42)

    # Return only the test data
    return test_data


@step
def validate_data(train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame) -> pd.DataFrame:
    """
    Step to validate data using TensorFlow Data Validation (TFDV).

    Args:
        train_data (pandas.DataFrame): The DataFrame containing the training data.
        test_data (pandas.DataFrame): The DataFrame containing the test data.

    Returns:
        pandas.DataFrame: The revised test_data DataFrame after handling anomalies.
    """
    # Generate statistics for training data
    train_stats = tfdv.generate_statistics_from_dataframe(train_data)
    # Display training data statistics
    tfdv.visualize_statistics(train_stats)

    # Infer a schema from the training data statistics
    schema = tfdv.infer_schema(statistics=train_stats)
    tfdv.display_schema(schema=schema)

    # Generate statistics for test data
    test_stats = tfdv.generate_statistics_from_dataframe(test_data)
    # Display test data statistics
    tfdv.visualize_statistics(test_stats)

    # Compare test statistics with the schema
    anomalies = tfdv.validate_statistics(statistics=test_stats, schema=schema)
    # Display anomalies
    tfdv.display_anomalies(anomalies)

    # Handle anomalies if they exist
    if anomalies.anomaly_info:
        print("Handling anomalies...")
        # Code to fix the anomalies goes here
        # Example: test_data['feature'] = test_data['feature'].clip(lower=0, upper=100)

        # Regenerate statistics and validate again after fixing
        test_stats = tfdv.generate_statistics_from_dataframe(test_data)
        anomalies = tfdv.validate_statistics(statistics=test_stats, schema=schema)

    # Return the revised test_data
    return test_data

# Tokenizer from BERT model
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM_PATH)
from transformers import BertTokenizer
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
from cassandra import ConsistencyLevel
from cassandra.policies import DCAwareRoundRobinPolicy
import uuid
# Feature Engineering step
@step
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Step for feature engineering using BERT tokenizer and storing features in Cassandra.

    Args:
        data (pd.DataFrame): Input DataFrame with text data.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Connect to Cassandra with load-balancing policy specified
    cluster = Cluster(contact_points=['127.0.0.1'], port=9042)
    session = cluster.connect()

    # Specify the keyspace
    session.set_keyspace('keyspace')

    # Create the 'features' table if it does not exist
    session.execute("""
        CREATE TABLE IF NOT EXISTS "keyspace".features (
            id UUID PRIMARY KEY,
            features list<int>
        )
    """)

    # Prepare a batch statement for inserting data into the 'features' table
    insert_statement = session.prepare('INSERT INTO "keyspace".features (id, features) VALUES (?, ?)')
    batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)
    # Add a 'tokens' column to the DataFrame if it doesn't exist
    if 'tokens' not in data.columns:
        data['tokens'] = None
    # Tokenize text data and store in Cassandra
    for index, row in data.iterrows():
        # Generate a UUID for each row
        row_id = uuid.uuid4()

        # Tokenize the text
        tokens = tokenizer.encode(row['text'], add_special_tokens=True)
        
        # Insert into Cassandra
        batch.add(insert_statement, (row_id, tokens))
        
        # Safely assign tokenized data to the 'tokens' column
        data.at[index, 'tokens'] = tokens

        # Execute batch every 2 records for efficiency
        if index % 2 == 0 and len(batch) > 0:
            session.execute(batch)
            batch.clear()

        # Execute any remaining statements
        if len(batch) > 0:
            session.execute(batch)

        # Close the Cassandra connection
        #session.shutdown()
        #cluster.shutdown()

    return data

from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import numpy as np
@step
def train_model(train_data: pd.DataFrame, val_data: pd.DataFrame, hyperparams: dict) -> BertForSequenceClassification:
    """
    Step to train the BERT model using PyTorch, with MLflow tracking.

    Args:
        train_data (pd.DataFrame): Training data.
        val_data (pd.DataFrame): Validation data.
        hyperparams (dict): Hyperparameters for training.

    Returns:
        BertForSequenceClassification: Trained BERT model.
    """
    # Load pre-trained BERT model for PyTorch
    unique_labels = train_data['label'].nunique()
    print(f"Number of unique labels: {unique_labels}")
    num_labels = len(np.unique(train_data['label'].values))
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_LM_PATH, num_labels=num_labels, ignore_mismatched_sizes=True)
    mlflow.log_dict(model.config.to_dict(), "model_config.json")
    print(f"Training on {len(train_data)} samples")
    print(f"Validating on {len(val_data)} samples")
    print(f"Training for {hyperparams['epochs']} epochs")
    print(f"Batch size: {hyperparams['batch_size']}")
    print(f"Learning rate: {hyperparams['learning_rate']}")
    mlflow.log_params(hyperparams)


    # Tokenization
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM_PATH)
    train_encodings = tokenizer(list(train_data['text']), truncation=True, padding=True, return_tensors="pt")
    val_encodings = tokenizer(list(val_data['text']), truncation=True, padding=True, return_tensors="pt")
    # Convert labels to LongTensor
    train_labels = torch.tensor(train_data['label'].values).long()
    val_labels = torch.tensor(val_data['label'].values).long()
    
   # Create TensorDataset
    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    val_dataset = TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], val_labels)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'])

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams['learning_rate'], weight_decay=hyperparams['weight_decay'])

    # Initialize the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hyperparams['lr_step_size'], gamma=hyperparams['lr_gamma'])


    # MLflow tracking   
    # End any existing run before starting a new one
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.start_run()
    mlflow.log_params(hyperparams)

    # Training loop
    # Checkpoint directory
    checkpoint_dir = "model_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float('inf')
    
    model.train()
    for epoch in range(hyperparams['epochs']):
        print(f"Starting epoch {epoch+1}/{hyperparams['epochs']}")
        total_loss = 0
        total_val_loss = 0
        print(f"Starting training for Epoch {epoch + 1}/{hyperparams['epochs']}")
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            total_val_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hyperparams['max_grad_norm'])
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Average training loss for epoch {epoch+1}: {avg_train_loss}")
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f"Adjusted learning rate to: {scheduler.get_last_lr()[0]}")
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate updated to {current_lr}")
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        
        # Validation loop
        model.eval()
        val_labels = []
        val_preds = []
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = outputs.loss
                if loss is not None:
                    total_val_loss += loss.item()
                val_labels.extend(labels.numpy())
                val_preds.extend(np.argmax(logits.numpy(), axis=1))
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Average validation loss for epoch {epoch+1}: {avg_val_loss}")
        scheduler.step(avg_val_loss)  # Adjust learning rate
        accuracy = accuracy_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds, average='macro')
        recall = recall_score(val_labels, val_preds, average='macro')
        f1 = f1_score(val_labels, val_preds, average='macro')

        # Log validation metrics
        print(f"Validation Metrics for epoch {epoch+1}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        mlflow.log_metrics({'val_accuracy': accuracy, 'val_precision': precision, 'val_recall': recall, 'val_f1': f1}, step=epoch)
        
        # Log the model to MLflow
        mlflow.pytorch.log_model(model, "model")
        mlflow.end_run()

    return model

# 7. Pipeline Definition
@pipeline
def biden_stance_pipeline(
    load_data_step,
    preprocess_data_step,
    visualize_data_step,
    split_data_1_step,
    split_data_2_step,
    split_data_3_step,
    validate_data_step,
    feature_engineering_step,
    train_model_step,

):
    data = load_data_step(csv_file_path=CSV_FILE_PATH)
    preprocessed_data = preprocess_data_step(data)
    visualize_data_step(preprocessed_data)
    train_data = split_data_1_step(data=preprocessed_data)
    val_data = split_data_2_step(data=preprocessed_data)
    test_data = split_data_3_step(data=preprocessed_data)
    validate_data_step(train_data, val_data, test_data)
    features = feature_engineering_step(train_data)
    model = train_model_step(train_data,val_data,HYPERPARAMS)

# Correctly instantiate the pipeline with step instances
biden_stance_pipeline = biden_stance_pipeline(
    load_data_step=load_data(),                 # Instantiate the load_data step
    preprocess_data_step=preprocess_data(),     # Instantiate the preprocess_data step
    visualize_data_step=visualize_data(),       # Instantiate the visualize_data step
    split_data_1_step=split_data_1(),               # Instantiate the split_data step
    split_data_2_step=split_data_2(),
    split_data_3_step=split_data_3(),
    validate_data_step=validate_data(),         # Instantiate the validate_data step
    feature_engineering_step = feature_engineering(),
    train_model_step=train_model()
)

# Run the pipeline
biden_stance_pipeline.run()
