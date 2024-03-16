# Imports (add necessary imports for your tools and libraries)
# General imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from transformers import BertTokenizer
import tensorflow as tf
# TensorFlow and related imports
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
import json
import requests
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import uuid
import json
import pandas as pd
from sklearn.model_selection import train_test_split



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
CSV_FILE_PATH = '/home/dino/Desktop/SP24/scripts/biden_stance_public.csv'
PRETRAINED_LM_PATH = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'
HYPERPARAMS = {
    "batch_size": 8,
    "learning_rate": 2e-5,
    "epochs": 2,
    "weight_decay": 0.01
}
print("Hyperparameters:", HYPERPARAMS)


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
    load_balancing_policy = DCAwareRoundRobinPolicy(local_dc='datacenter1')
    cluster = Cluster(contact_points=['127.0.0.1'], load_balancing_policy=load_balancing_policy)
    session = cluster.connect()

    # Specify the keyspace
    session.set_keyspace('my_keyspace')

    # Create the 'features' table if it does not exist
    session.execute("""
        CREATE TABLE IF NOT EXISTS my_keyspace.features (
            id UUID PRIMARY KEY,
            features list<int>
        )
    """)

    # Prepare a batch statement for inserting data into the 'features' table
    insert_statement = session.prepare("INSERT INTO my_keyspace.features (id, features) VALUES (?, ?)")
    batch = BatchStatement(consistency_level=ConsistencyLevel.QUORUM)
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

# 7. Pipeline Definition
@pipeline
def stance_pipeline(
    load_data_step,
    preprocess_data_step,
    visualize_data_step,
    split_data_1_step,
    split_data_2_step,
    split_data_3_step,
    validate_data_step,
    feature_engineering_step,

):
    data = load_data_step(csv_file_path=CSV_FILE_PATH)
    preprocessed_data = preprocess_data_step(data)
    visualize_data_step(preprocessed_data)
    train_data = split_data_1_step(data=preprocessed_data)
    val_data = split_data_2_step(data=preprocessed_data)
    test_data = split_data_3_step(data=preprocessed_data)
    validate_data_step(train_data, val_data, test_data)
    features = feature_engineering_step(train_data)

# Correctly instantiate the pipeline with step instances
stance_pipeline = stance_pipeline(
    load_data_step=load_data(),                 # Instantiate the load_data step
    preprocess_data_step=preprocess_data(),     # Instantiate the preprocess_data step
    visualize_data_step=visualize_data(),       # Instantiate the visualize_data step
    split_data_1_step=split_data_1(),               # Instantiate the split_data step
    split_data_2_step=split_data_2(),  
    split_data_3_step=split_data_3(),  
    validate_data_step=validate_data(),         # Instantiate the validate_data step
    feature_engineering_step = feature_engineering()
)

# Run the pipeline
stance_pipeline.run()


