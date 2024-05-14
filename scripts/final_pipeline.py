from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module=".*tensorflow.*")
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import numpy as np
import os
import logging
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from zenml.steps import step
from zenml.pipelines import pipeline
import tweepy
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
import pandas as pd
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_data_validation as tfdv
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore')
from transformers import BertTokenizer
from cassandra.cluster import Cluster
from cassandra.query import BatchStatement
from cassandra import ConsistencyLevel
import uuid
import pandas as pd
import logging
from cassandra.policies import DCAwareRoundRobinPolicy
import pytest

#Final pipeline until training




# Constants and Hyperparameters
CSV_FILE_PATH = 'C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\dataset_reduced.csv'
PRETRAINED_LM_PATH = 'C:\\Users\\LENOVO\\Desktop\\bert-election2024-twitter-stance-biden'
HYPERPARAMS = {
    "batch_size": 4,             # Batch size for training
    "learning_rate": 1e-5,       # Learning rate for the optimizer
    "epochs": 10,                # Number of training epochs
    "max_length": 150,           # Max length for tokenized sequences
    "num_labels": 3,             # Number of output labels for classification
    "ignore_mismatched_sizes": True,  # To ignore size mismatches in the model
    "optimizer_eps": 1e-8,       # Epsilon value for the AdamW optimizer
    "num_warmup_steps": 0,       # Number of warmup steps for the schedulerHYPERPARAMS = {
}
print("Hyperparameters:", HYPERPARAMS)
mlflow.set_tracking_uri('http://127.0.0.1:5000')
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM_PATH)


@step
def load_data(csv_file_path):
    """
    Step to load data from a CSV file. This function includes error handling,
    logging, and a greeting message for a user-friendly interface. The approach
    is inspired by the meticulous data handling seen in the stance detection paper.

    Args:
    csv_file_path (str): Path to the CSV file.

    Returns:
    pandas.DataFrame: Loaded data or an informative error message.
    """
    try:
        print("Greetings! Initiating the data loading process.")
        # Load the dataset
        data = pd.read_csv(csv_file_path)
        #Reset index
        data.set_index('tweet_id', inplace = True)
        # Log successful loading
        logging.info(f"Data successfully loaded from {csv_file_path}.")
        #Preview
        data.head()
        print(f"Data from {csv_file_path} loaded successfully. The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")

        # Perform initial checks similar to the paper's approach, e.g., checking for null values
        if data.isnull().values.any():
            null_counts = data.isnull().sum()
            print("Warning: Null values found in the dataset.")
            print(f"Null value counts by column:\n{null_counts[null_counts > 0]}")

        # Return the loaded data
        return data

    except FileNotFoundError:
        logging.error(f"File not found: {csv_file_path}")
        print(f"Error: The file {csv_file_path} was not found. Please check the file path.")

    except pd.errors.ParserError:
        logging.error(f"Error parsing the file: {csv_file_path}")
        print(f"Error: Could not parse the file {csv_file_path}. Please check if the file is a valid CSV.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An unexpected error occurred: {e}")



@step
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Step for data preprocessing, including cleaning, tokenization, and label encoding.

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
        # Removing usernames and hashtags
        text = re.sub(r'@\S+|#\S+', '', text)
        # Removing special characters and numbers
        text = re.sub(r'[^A-Za-z\s]', '', text)
        # Converting to lowercase
        text = text.lower().strip()
        # Tokenize and rejoin the text to ensure clean tokenization
        tokens = text.split()
        return ' '.join(tokens)

    # Ensure the dataframe has the expected columns
    expected_columns = {'text','label'}
    data.info()
    data.isnull().sum()
    if not expected_columns.issubset(data.columns):
        missing_cols = expected_columns - set(data.columns)
        error_msg = f"The dataframe is missing the following required columns: {', '.join(missing_cols)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    # Clean text data
    #data['text'] = data['text'].astype(str).apply(clean_text)

    print("Preprocessing the data...")
    data.text.iloc[10]
    data.label.value_counts()
    missing_label_rows = data[data['label'].isna()]
    data.dropna(subset=['label'], inplace=True)

    if not missing_label_rows.empty:
        print(f"Rows with missing labels:\n{missing_label_rows}")
    # If the DataFrame is empty after dropping missing values, return it as is
    if data.empty:
        print("The DataFrame is empty after preprocessing.")
        return data
    # Encode labels
    label_mapping = {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2}
    data['label'] = data['label'].map(label_mapping)

    # Validate the encoding
    unique_labels = data['label'].unique()
    if set(unique_labels) != {0, 1, 2}:
        error_msg = f"Labels are not correctly mapped. Found unique labels: {unique_labels}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    print("Data preprocessing complete. Text has been cleaned and labels encoded.")

    return data


@step
def visualize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Step for data visualization including checking for missing values and data distribution.

    Args:
        data (pandas.DataFrame): The preprocessed data.

    Returns:
        pandas.DataFrame: The data (if the next step in the pipeline requires it).
    """
    print("Starting data visualization process...")

    try:
        # Visualize Missing Values
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
        logging.error(f"Error occurred during missing values visualization: {e}")
        print(f"Error occurred during missing values visualization: {e}")

    try:
        # Data Distribution Visualization
        if 'label' in data.columns:
            plt.figure(figsize=(8, 4))
            sns.countplot(x='label', data=data)
            plt.title('Distribution of Labels')
            plt.xlabel('Label')
            plt.ylabel('Count')
            plt.show()
        else:
            print("Label column not found, skipping label distribution visualization.")
        
    except Exception as e:
        logging.error(f"Error occurred during label distribution visualization: {e}")
        print(f"Error occurred during label distribution visualization: {e}")

    print("Data visualization process completed.")
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
        val_data (pandas.DataFrame): The DataFrame containing the validation data.
        test_data (pandas.DataFrame): The DataFrame containing the test data.

    Returns:
        pandas.DataFrame: The revised test_data DataFrame after handling anomalies, if found.
    """
    # Generate statistics for training, validation, and test data
    try:
        train_stats = tfdv.generate_statistics_from_dataframe(train_data)
        val_stats = tfdv.generate_statistics_from_dataframe(val_data)
        test_stats = tfdv.generate_statistics_from_dataframe(test_data)
    except Exception as e:
        logging.error(f"Error in statistics generation: {e}")
        raise e

    print("Statistics generated for Train, Validation, and Test datasets.")

    # Infer schema from training data
    schema = tfdv.infer_schema(statistics=test_stats)
    print("Schema inferred from the testing data.")

    # Validate validation and test data using the inferred schema
    try:
        for dataset_name, dataset_stats in [('Validation', val_stats), ('Test', test_stats)]:
            print(f"Validating {dataset_name} data...")
            anomalies = tfdv.validate_statistics(statistics=dataset_stats, schema=schema)
            
            if anomalies.anomaly_info:
                print(f"Anomalies found in {dataset_name} data:")
                tfdv.display_anomalies(anomalies)
                # Potential handling or correction of anomalies
                # E.g., aligning categorical features, handling missing values, etc.
                # This will depend on the nature of anomalies detected
            else:
                print(f"No anomalies detected in {dataset_name} data.")
    except Exception as e:
        logging.error(f"Error during data validation: {e}")
        raise e

    # Display schema for review
    print("Displaying the inferred schema:")
    tfdv.display_schema(schema=schema)

    return test_data

import random

@step
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Step for feature engineering using BERT tokenizer and storing features in Cassandra.
    Adds error handling, logging, and user-friendly print statements.

    Args:
        data (pd.DataFrame): Input DataFrame with text data.

    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    try:
        print("Connecting to the Cassandra Database...")
        cluster = Cluster(contact_points=['127.0.0.1'], port=9042, load_balancing_policy=DCAwareRoundRobinPolicy())
        session = cluster.connect()
        session.set_keyspace('keyspace_1')
        session.execute("""
            CREATE TABLE IF NOT EXISTS "keyspace_1".features (
                id UUID PRIMARY KEY,
                features list<int>
            )
        """)

        insert_statement = session.prepare('INSERT INTO keyspace_1.features (id, features) VALUES (?, ?)')
        batch = BatchStatement(consistency_level=ConsistencyLevel.ONE)
        batch_size_limit = 8

        # Initialize 'tokens' column as a list of lists if it does not exist
        if 'tokens' not in data.columns:
            data['tokens'] = [[] for _ in range(len(data))]

        for index, row in data.iterrows():
            row_id = uuid.uuid4()
            # Example tokenizer function call; ensure it returns a list of integers
            tokens = tokenizer.encode(row['text'], add_special_tokens=True)
            # Set the tokens in the DataFrame
            data.at[index, 'tokens'] = tokens
            # Add to batch using correct parameters
            batch.add(insert_statement, row_id, tokens)

            if len(batch) >= batch_size_limit:
                session.execute(batch)
                batch.clear()

        if len(batch) > 0:
            session.execute(batch)

        print("Feature engineering and data insertion completed successfully.")
        return data


    except Exception as e:
        logging.error(f"Error during feature engineering: {e}")
        print(f"An error occurred during feature engineering: {e}")
        # Optionally, re-raise the exception if you want the step to fail in case of errors
        raise e

    finally:
        # Ensure the Cassandra connection is always closed
        session.shutdown()
        cluster.shutdown()
def compute_confusion_matrix(true_labels, predictions):
    """
    Computes the confusion matrix.
    
    Args:
    true_labels (array): Array of true labels.
    predictions (array): Array of model predictions.

    Returns:
    ndarray: Confusion matrix.
    """
    return confusion_matrix(true_labels, predictions)

def compute_accuracy(true_labels, predictions):
    """
    Computes the accuracy score.
    
    Args:
    true_labels (array): Array of true labels.
    predictions (array): Array of model predictions.

    Returns:
    float: Accuracy score.
    """
    return accuracy_score(true_labels, predictions)

def compute_recall(true_labels, predictions):
    """
    Computes the recall score.
    
    Args:
    true_labels (array): Array of true labels.
    predictions (array): Array of model predictions.

    Returns:
    float: Recall score.
    """
    return recall_score(true_labels, predictions, average='macro')

def compute_precision(true_labels, predictions):
    """
    Computes the precision score.
    
    Args:
    true_labels (array): Array of true labels.
    predictions (array): Array of model predictions.

    Returns:
    float: Precision score.
    """
    return precision_score(true_labels, predictions, average='macro')

def evaluate_model(model, data_loader, device):
    model.eval()
    val_labels = []
    val_preds = []
    with torch.no_grad():
        try:
            for batch in data_loader:
                # Ensure all tensors are on the same device
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(torch.argmax(logits, axis=1).cpu().numpy())
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")
            return None

    # Compute detailed metrics
    report = classification_report(val_labels, val_preds, output_dict=True)
    f1 = report['macro avg']['f1-score']
    cm = compute_confusion_matrix(val_labels, val_preds)
    accuracy = compute_accuracy(val_labels, val_preds)
    recall = compute_recall(val_labels, val_preds)
    precision = compute_precision(val_labels, val_preds)

    # Include metrics for each class if necessary
    detailed_metrics = {f'class_{k}': v for k, v in report.items() if k.isdigit()}

    return {
        'confusion_matrix': cm, 
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
        'f1': f1,
        'detailed_metrics': detailed_metrics
    }


from tqdm.notebook import tqdm
from sklearn.utils.class_weight import compute_class_weight

@step
def train_model(data: pd.DataFrame,train_data: pd.DataFrame, val_data: pd.DataFrame, hyperparams: dict) -> BertForSequenceClassification:
    """
    Step to train the BERT model using PyTorch, with MLflow tracking.
    Includes error handling, logging, and detailed progress updates.

    Args:
        data
        train_data (pd.DataFrame): Training data.
        val_data (pd.DataFrame): Validation data.
        hyperparams (dict): Hyperparameters for training.

    Returns:
        BertForSequenceClassification: Trained BERT model.
    """
    PRETRAINED_LM_PATH = 'C:\\Users\\LENOVO\\Desktop\\SP24\\bert-election2020-twitter-stance-biden'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use a temporary directory for checkpoints
    checkpoint_dir = os.path.join(os.getcwd(), "C:\\Users\\LENOVO\\Desktop\\SP24\\scripts\\model_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    def check_labels(data):
        unique_labels = np.unique(data['label'])
        if not np.array_equal(unique_labels, [0, 1, 2]):
            raise ValueError(f"Labels out of bounds. Expected labels [0, 1, 2], found {unique_labels}")
    expected_columns = {'text','label'}
    if not expected_columns.issubset(data.columns):
        missing_cols = expected_columns - set(data.columns)
        error_msg = f"The dataframe is missing the following required columns: {', '.join(missing_cols)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    missing_label_rows = data[data['label'].isna()]
    data.dropna(subset=['label'], inplace=True)
    if not missing_label_rows.empty:
        print(f"Rows with missing labels:\n{missing_label_rows}")
    # If the DataFrame is empty after dropping missing values, return it as is
    if data.empty:
        print("The DataFrame is empty after preprocessing.")
    # Encode labels
    label_mapping = {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2}
    data['label'] = data['label'].map(label_mapping)
    data.head(15)
    data['data_type'] = ['not_set'] * data.shape[0]
    X_train, X_val, y_train, y_val = train_test_split(data.index.values, data.label.values,test_size = 0.15,random_state = 17,stratify = data.label.values)
    data.loc[X_train, 'data_type'] = 'train'
    data.loc[X_val, 'data_type'] = 'val'
    data.groupby(['label', 'data_type']).count()
    #Tokenize train set
    encoded_data_train = tokenizer.batch_encode_plus(data[data.data_type == 'train'].text.values,
                                                add_special_tokens = True,
                                                return_attention_mask = True,
                                                pad_to_max_length = True,
                                                max_length = 150,
                                                return_tensors = 'pt')
    #Tokenizer val set
    encoded_data_val = tokenizer.batch_encode_plus(data[data.data_type == 'val'].text.values,
                                                #add_special_tokens = True,
                                                return_attention_mask = True,
                                                pad_to_max_length = True,
                                                max_length = 150,
                                                return_tensors = 'pt')
    encoded_data_train
    #Encode train set
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(data[data.data_type == 'train'].label.values)
    #Encode val set
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']

    #Convert data type to torch.tensor
    labels_val = torch.tensor(data[data.data_type == 'val'].label.values)

    #Create dataloader
    dataset_train = TensorDataset(input_ids_train, attention_masks_train,labels_train)

    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    print(len(dataset_train))
    print(len(dataset_val))

    model = BertForSequenceClassification.from_pretrained(PRETRAINED_LM_PATH, num_labels=3, ignore_mismatched_sizes=True)
    model.config

    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

    batch_size = 4 #Since we have limited resource

    #Load train set
    dataloader_train = DataLoader(dataset_train,sampler = RandomSampler(dataset_train),batch_size = batch_size)

    #Load val set
    dataloader_val = DataLoader(dataset_val,sampler = RandomSampler(dataset_val),batch_size = 32) #since we don't have to do backpropagation for this step

    from transformers import AdamW, get_linear_schedule_with_warmup
    epochs = 10

    #Load optimizer
    optimizer = AdamW(model.parameters(),lr = 1e-5,eps = 1e-8) #2e-5 > 5e-5

    #Load scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps = 0,num_training_steps = len(dataloader_train)*epochs)

    #F1 score
    def f1_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average = 'weighted')
    
    #Accuracy score
    def accuracy_per_class(preds, labels):
        label_dict_inverse = {v: k for k, v in label_dict.items()}
        
        #Make prediction
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        
        for label in np.unique(labels_flat):
            y_preds = preds_flat[labels_flat==label]
            y_true = labels_flat[labels_flat==label]
            print(f'Class: {label_dict_inverse[label]}')
            print(f'Accuracy:{len(y_preds[y_preds==label])}/{len(y_true)}\n')

    def evaluate(dataloader_val):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #Evaluation mode disables the dropout layer 
        model.eval()
        
        #Tracking variables
        loss_val_total = 0
        predictions, true_vals = [], []
        
        for batch in tqdm(dataloader_val):
            
            #Load into GPU
            batch = tuple(b.to(device) for b in batch)
            
            #Define inputs
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2]}

            #Compute logits
            with torch.no_grad():        
                outputs = model(**inputs)
            
            #Compute loss
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            #Compute accuracy
            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)
            
        #Compute average loss
        loss_val_avg = loss_val_total/len(dataloader_val) 
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
        return loss_val_avg, predictions, true_vals

    seed_val = 17
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    try:
        with mlflow.start_run():
            print("Initializing the training process...")
            check_labels(train_data)
            check_labels(val_data)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", 1e-5)
            mlflow.log_param("epochs", epochs)

            model = BertForSequenceClassification.from_pretrained(PRETRAINED_LM_PATH, num_labels=3, ignore_mismatched_sizes=True)

            # Progress bar for epochs
            epoch_progress = tqdm(range(1, epochs+1), desc="Training Progress", leave=True)
            # Calculate class weights for handling class imbalance
            class_weights = compute_class_weight(class_weight='balanced', 
                                                classes=np.unique(train_data['label']), 
                                                y=train_data['label'])
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

            # Initialize the optimizer with a lower learning rate
            optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader_train)*epochs)

            # Implement early stopping
            best_val_loss = float('inf')
            no_improvement_epochs = 0
            early_stopping_threshold = 3  # stop training if no improvement for 3 epochs

            for epoch in epoch_progress:
                model.train()
                loss_train_total = 0

                # Nested progress bar for training batches
                batch_progress = tqdm(dataloader_train, desc=f'Epoch {epoch}/{epochs}', leave=False)
                for batch in batch_progress:
                    model.zero_grad()
                    batch = tuple(b.to(device) for b in batch)
                    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                    outputs = model(**inputs)
                    loss = outputs[0]
                    loss_train_total += loss.item()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    # Update progress bar for batches
                    batch_progress.set_postfix({'Batch Loss': '{:.3f}'.format(loss.item()/len(batch))})

                # Calculate and display epoch metrics
                loss_train_avg = loss_train_total/len(dataloader_train)
                epoch_progress.set_postfix({'Epoch Avg Loss': '{:.3f}'.format(loss_train_avg)})

                val_loss, predictions, true_vals = evaluate(dataloader_val)
                val_f1 = f1_score_func(predictions, true_vals)

                # Log metrics in tqdm and MLflow
                tqdm.write(f'Epoch {epoch}/{epochs} - Training Loss: {loss_train_avg}, Validation Loss: {val_loss}, F1 Score: {val_f1}')
                mlflow.log_metric("training_loss", loss_train_avg, step=epoch)
                mlflow.log_metric("validation_loss", val_loss, step=epoch)
                mlflow.log_metric("F1_score", val_f1, step=epoch)
                # Saving the model checkpoint after each epoch
                checkpoint_dir = "C:\\Users\\LENOVO\\Desktop\\SP24\\scripts\\model_checkpoints"
                epoch_model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
                torch.save(model.state_dict(), epoch_model_path)
                # Log the checkpoint in MLflow
                mlflow.log_artifact(epoch_model_path)
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improvement_epochs = 0
                else:
                    no_improvement_epochs += 1
                    if no_improvement_epochs >= early_stopping_threshold:
                        print(f"Early stopping triggered at epoch {epoch}")
                        torch.save(model.state_dict(), epoch_model_path)
                        mlflow.log_artifact("C:\\Users\\LENOVO\\Desktop\\SP24\\scripts\\model_checkpoints")
                        break
                torch.save(model.state_dict(), epoch_model_path)
                mlflow.log_artifact("C:\\Users\\LENOVO\\Desktop\\SP24\\scripts\\model_checkpoints")

    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        print(f"An error occurred during model training: {e}")
        mlflow.end_run()
        raise e

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
    train_data = validate_data_step(train_data, val_data, test_data)
    features = feature_engineering_step(train_data)
    model = train_model_step(data,train_data,val_data,HYPERPARAMS)

# Correctly instantiate the pipeline with step instances
biden_stance_pipeline = biden_stance_pipeline(
    load_data_step=load_data(),
    preprocess_data_step=preprocess_data(),
    visualize_data_step=visualize_data(),
    split_data_1_step=split_data_1(),
    split_data_2_step=split_data_2(),
    split_data_3_step=split_data_3(),
    validate_data_step=validate_data(),
    feature_engineering_step = feature_engineering(),
    train_model_step=train_model()
)

# Run the pipeline
biden_stance_pipeline.run()
