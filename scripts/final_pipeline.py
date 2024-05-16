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



# Constants and Hyperparameters
CSV_FILE_PATH = 'C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\dataset_reduced.csv'
PRETRAINED_LM_PATH = 'C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\bert-election2024-twitter-stance-biden'
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
def load_data(csv_file_path: str) -> pd.DataFrame:
    try:
        print("Greetings! Initiating the data loading process.")
        data = pd.read_csv(csv_file_path)
        data.set_index('tweet_id', inplace=True)
        logging.info(f"Data successfully loaded from {csv_file_path}.")
        data.head()
        print(f"Data from {csv_file_path} loaded successfully. The dataset contains {data.shape[0]} rows and {data.shape[1]} columns.")
        if data.isnull().values.any():
            null_counts = data.isnull().sum()
            print("Warning: Null values found in the dataset.")
            print(f"Null value counts by column:\n{null_counts[null_counts > 0]}")
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
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\S+|#\S+', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower().strip()
        tokens = text.split()
        return ' '.join(tokens)

    expected_columns = {'text', 'label'}
    data.info()
    data.isnull().sum()
    if not expected_columns.issubset(data.columns):
        missing_cols = expected_columns - set(data.columns)
        error_msg = f"The dataframe is missing the following required columns: {', '.join(missing_cols)}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    print("Preprocessing the data...")
    data['text'] = data['text'].astype(str).apply(clean_text)
    data.text.iloc[10]
    data.label.value_counts()
    missing_label_rows = data[data['label'].isna()]
    data.dropna(subset=['label'], inplace=True)
    if not missing_label_rows.empty:
        print(f"Rows with missing labels:\n{missing_label_rows}")
    if data.empty:
        print("The DataFrame is empty after preprocessing.")
        return data
    label_mapping = {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2}
    data['label'] = data['label'].map(label_mapping)
    unique_labels = data['label'].unique()
    if set(unique_labels) != {0, 1, 2}:
        error_msg = f"Labels are not correctly mapped. Found unique labels: {unique_labels}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    print("Data preprocessing complete. Text has been cleaned and labels encoded.")
    return data

@step
def split_data_1(data: pd.DataFrame) -> pd.DataFrame:
    test_size = 0.2
    validation_size = 0.1
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    adjusted_validation_size = validation_size / (1 - test_size)
    train_data, val_data = train_test_split(train_val_data, test_size=adjusted_validation_size, random_state=42)
    return train_data

@step
def split_data_2(data: pd.DataFrame) -> pd.DataFrame:
    test_size = 0.2
    validation_size = 0.1
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    adjusted_validation_size = validation_size / (1 - test_size)
    train_data, val_data = train_test_split(train_val_data, test_size=adjusted_validation_size, random_state=42)
    return val_data

@step
def split_data_3(data: pd.DataFrame) -> pd.DataFrame:
    test_size = 0.2
    validation_size = 0.1
    train_val_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    adjusted_validation_size = validation_size / (1 - test_size)
    train_data, val_data = train_test_split(train_val_data, test_size=adjusted_validation_size, random_state=42)
    return test_data

@step
def train_model(train_data: pd.DataFrame, val_data: pd.DataFrame, hyperparams: dict) -> BertForSequenceClassification:
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def check_labels(data):
        unique_labels = np.unique(data['label'])
        if not np.array_equal(unique_labels, [0, 1, 2]):
            raise ValueError(f"Labels out of bounds. Expected labels [0, 1, 2], found {unique_labels}")

    check_labels(train_data)
    check_labels(val_data)

    data = train_data.copy()
    data['data_type'] = ['not_set'] * data.shape[0]
    X_train, X_val, y_train, y_val = train_test_split(data.index.values, data.label.values, test_size=0.15, random_state=17, stratify=data.label.values)
    data.loc[X_train, 'data_type'] = 'train'
    data.loc[X_val, 'data_type'] = 'val'
    data.groupby(['label', 'data_type']).count()
    
    encoded_data_train = tokenizer.batch_encode_plus(data[data.data_type == 'train'].text.values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=150, return_tensors='pt')
    encoded_data_val = tokenizer.batch_encode_plus(data[data.data_type == 'val'].text.values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=150, return_tensors='pt')
    
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']
    labels_train = torch.tensor(data[data.data_type == 'train'].label.values)
    input_ids_val = encoded_data_val['input_ids']
    attention_masks_val = encoded_data_val['attention_mask']
    labels_val = torch.tensor(data[data.data_type == 'val'].label.values)

    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    model = BertForSequenceClassification.from_pretrained(PRETRAINED_LM_PATH, num_labels=3, ignore_mismatched_sizes=True)
    batch_size = hyperparams["batch_size"]
    dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)
    dataloader_val = DataLoader(dataset_val, sampler=RandomSampler(dataset_val), batch_size=32)
    optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"], eps=hyperparams["optimizer_eps"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=hyperparams["num_warmup_steps"], num_training_steps=len(dataloader_train) * hyperparams["epochs"])

    def f1_score_func(preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def evaluate(dataloader_val):
        model.eval()
        loss_val_total = 0
        predictions, true_vals = [], []
        with torch.no_grad():
            for batch in dataloader_val:
                batch = tuple(b.to(device) for b in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = model(**inputs)
                loss = outputs[0]
                logits = outputs[1]
                loss_val_total += loss.item()
                logits = logits.detach().cpu().numpy()
                label_ids = inputs['labels'].cpu().numpy()
                predictions.append(logits)
                true_vals.append(label_ids)
        loss_val_avg = loss_val_total / len(dataloader_val)
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
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", hyperparams["learning_rate"])
            mlflow.log_param("epochs", hyperparams["epochs"])
            epoch_progress = tqdm(range(1, hyperparams["epochs"] + 1), desc="Training Progress", leave=True)
            best_val_loss = float('inf')
            no_improvement_epochs = 0
            early_stopping_threshold = 3

            for epoch in epoch_progress:
                model.train()
                loss_train_total = 0
                batch_progress = tqdm(dataloader_train, desc=f'Epoch {epoch}/{hyperparams["epochs"]}', leave=False)
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
                    batch_progress.set_postfix({'Batch Loss': '{:.3f}'.format(loss.item() / len(batch))})
                loss_train_avg = loss_train_total / len(dataloader_train)
                epoch_progress.set_postfix({'Epoch Avg Loss': '{:.3f}'.format(loss_train_avg)})
                val_loss, predictions, true_vals = evaluate(dataloader_val)
                val_f1 = f1_score_func(predictions, true_vals)
                tqdm.write(f'Epoch {epoch}/{hyperparams["epochs"]} - Training Loss: {loss_train_avg}, Validation Loss: {val_loss}, F1 Score: {val_f1}')
                mlflow.log_metric("training_loss", loss_train_avg, step=epoch)
                mlflow.log_metric("validation_loss", val_loss, step=epoch)
                mlflow.log_metric("F1_score", val_f1, step=epoch)
                checkpoint_dir = "C:\\Users\\LENOVO\\Desktop\\SP24\\scripts\\model_checkpoints"
                epoch_model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
                torch.save(model.state_dict(), epoch_model_path)
                mlflow.log_artifact(epoch_model_path)
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

@step
def model_inference(model: BertForSequenceClassification, data: pd.DataFrame, hyperparams: dict) -> pd.DataFrame:
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_LM_PATH)
    encoded_data_test = tokenizer.batch_encode_plus(data['text'].values, add_special_tokens=True, return_attention_mask=True, pad_to_max_length=True, max_length=hyperparams['max_length'], return_tensors='pt')
    input_ids_test = encoded_data_test['input_ids']
    attention_masks_test = encoded_data_test['attention_mask']
    labels_test = torch.tensor(data['label'].values)
    dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)
    dataloader_test = DataLoader(dataset_test, batch_size=hyperparams['batch_size'], sampler=SequentialSampler(dataset_test))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in dataloader_test:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            predictions.append(torch.argmax(logits, axis=1).cpu().numpy())
            true_labels.append(b_labels.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    return pd.DataFrame({'true_labels': true_labels, 'predicted_labels': predictions})

@step
def model_evaluation(predictions: pd.DataFrame) -> dict:
    true_labels = predictions['true_labels'].values
    predicted_labels = predictions['predicted_labels'].values
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    cm = confusion_matrix(true_labels, predicted_labels)
    evaluation_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()  
    }
    print("Evaluation Metrics: ", evaluation_metrics)
    return evaluation_metrics

@pipeline
def biden_stance_pipeline(
    load_data_step,
    preprocess_data_step,
    split_data_1_step,
    split_data_2_step,
    split_data_3_step,
    train_model_step,
    model_inference_step,
    model_evaluation_step
):
    data = load_data_step(csv_file_path=CSV_FILE_PATH)
    preprocessed_data = preprocess_data_step(data)
    train_data = split_data_1_step(data=preprocessed_data)
    val_data = split_data_2_step(data=preprocessed_data)
    test_data = split_data_3_step(data=preprocessed_data)
    model = train_model_step(train_data=train_data, val_data=val_data, hyperparams=HYPERPARAMS)
    predictions = model_inference_step(model=model, data=test_data, hyperparams=HYPERPARAMS)
    evaluation_metrics = model_evaluation_step(predictions=predictions)

biden_stance_pipeline = biden_stance_pipeline(
    load_data_step=load_data(),
    preprocess_data_step=preprocess_data(),
    split_data_1_step=split_data_1(),
    split_data_2_step=split_data_2(),
    split_data_3_step=split_data_3(),
    train_model_step=train_model(),
    model_inference_step=model_inference(),
    model_evaluation_step=model_evaluation()
)

biden_stance_pipeline.run()