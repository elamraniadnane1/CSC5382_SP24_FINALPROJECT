import mlflow
import mlflow.pytorch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
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


print("Welcome to the Stance Detection Model Training!")


# Define the path to the model files and the CSV file
CSV_FILE_PATH = '/home/dino/Desktop/SP24/scripts/biden_new_2024.csv'
pretrained_LM_path = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'

def validate_and_fix_data(df):
    stats = tfdv.generate_statistics_from_dataframe(df)
    schema = tfdv.infer_schema(stats)
    anomalies = tfdv.validate_statistics(stats, schema)
    print("Initial Dataset Information:")
    print("Shape:", df.shape)
    print("Sample Data:\n", df.head())
    print("Basic Statistics:\n", df.describe())

    # Display and log anomalies
    if anomalies.anomaly_info:
        print("Anomalies detected:")
        for feature_name, anomaly in anomalies.anomaly_info.items():
            print(f"Feature: {feature_name} - Anomaly: {anomaly.description}")
        # Consider fixing anomalies here based on your specific requirements
    else:
        print("No anomalies detected in the dataset.")
    
    # Log schema and anomalies with MLflow
    mlflow.log_dict(schema.as_proto().SerializeToString(), "schema.pbtxt")
    mlflow.log_dict(anomalies.SerializeToString(), "anomalies.pbtxt")
    
    # Return the modified dataframe
    return df
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

# Add and commit changes
def commit_changes(commit_message):
    run_command("git add .")
    run_command(f"git commit -m '{commit_message}'")

# Push changes to remote repository
def push_to_remote(remote_name, branch_name):
    run_command(f"git push {remote_name} {branch_name}")



def generate_statistics(df):
    stats = tfdv.generate_statistics_from_dataframe(df)
    return stats



def infer_and_validate_schema(df):
    # Check for required columns
    required_columns = ['text', 'mapped_label']  # Updated to 'mapped_label'
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    # Check data types
    expected_types = {'text': str, 'mapped_label': int}  # Updated to 'mapped_label'
    for col, expected_type in expected_types.items():
        if not all(isinstance(x, expected_type) for x in df[col].dropna()):
            raise ValueError(f"Column {col} has incorrect data type")

    # Check for missing values in essential columns
    if df[required_columns].isnull().any().any():
        raise ValueError("Missing values found in essential columns")

    print("Schema validation passed.")


def compute_statistics_and_detect_anomalies(df):
    # Assuming 'mapped_label' is the numerical representation of the 'label' column
    label_stats = df['mapped_label'].describe()
    print("Statistics for 'mapped_label':\n", label_stats)
    print("Anomaly handling steps:")
    print("- Converting 'text' to string.")
    print("- Dropping missing values.")
    print("- Removing outliers based on IQR.")

    # Detecting outliers using IQR
    Q1 = df['mapped_label'].quantile(0.25)
    Q3 = df['mapped_label'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['mapped_label'] < (Q1 - 1.5 * IQR)) | (df['mapped_label'] > (Q3 + 1.5 * IQR))]
    if not outliers.empty:
        print("Outliers detected in 'mapped_label':\n", outliers)

def infer_schema(stats):
    schema = tfdv.infer_schema(statistics=stats)
    return schema

def validate_dataset(df, schema):
    anomalies = tfdv.validate_statistics(statistics=generate_statistics(df), schema=schema)
    return anomalies


def handle_schema_and_anomalies(df):
    # Fixing any incorrect data types
    df['text'] = df['text'].astype(str)

    # Handling missing values
    df.dropna(subset=['text', 'mapped_label'], inplace=True)

    # Removing outliers from the 'mapped_label' column
    Q1 = df['mapped_label'].quantile(0.25)
    Q3 = df['mapped_label'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df['mapped_label'] < (Q1 - 1.5 * IQR)) | (df['mapped_label'] > (Q3 + 1.5 * IQR)))]

    print("Schema revisions and anomalies handled.")




# Set hyperparameters
hyperparams = {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "epochs": 4,
    "weight_decay": 0.01
}

print("Hyperparameters:", hyperparams)

# Start MLflow run
with mlflow.start_run() as run:
    print("MLflow run started. Logging hyperparameters.")
    mlflow.log_params(hyperparams)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)
    mlflow.log_dict(model.config.to_dict(), "model_config.json")

    id2label = {0: "AGAINST", 1: "FAVOR", 2: "NONE"}
    label2id = {v: k for k, v in id2label.items()}

    print("Reading and processing the dataset...")
    df = pd.read_csv(CSV_FILE_PATH)
    

    # Generate statistics using TensorFlow Data Validation
    stats = generate_statistics(df)

    # Infer schema from the generated statistics
    schema = infer_schema(stats)

    # Validate the dataset using the inferred schema
    anomalies = validate_dataset(df, schema)

    # Display any detected anomalies
    if anomalies.anomaly_info:
        print("Anomalies detected:")
        for feature_name, anomaly in anomalies.anomaly_info.items():
            print(f"Feature: {feature_name} - Anomaly: {anomaly.description}")
    else:
        print("No anomalies detected in the dataset.")

    print("Unique values in 'label' column:", df['label'].unique())

    # Define label mapping
    label2id = {'AGAINST': 0, 'FAVOR': 1, 'NONE': 2}

    # Apply label mapping
    df['mapped_label'] = df['label'].map(label2id)

    # Check for unmapped labels
    if df['mapped_label'].isnull().any():
        missing_labels = df[df['mapped_label'].isnull()]['label'].unique()
        raise ValueError(f"Found unmapped label values: {missing_labels}")

    # Debugging: print out a few mapped labels
    print("Sample mapped labels:", df['mapped_label'].head())

    # Convert mapped labels to integers
    df['mapped_label'] = df['mapped_label'].astype(int)

    # Ensure all labels are valid
    mapped_labels = df['mapped_label'].tolist()
    if not all(isinstance(label, int) and label in label2id.values() for label in mapped_labels):
        raise ValueError("Found invalid label values")

    # Convert labels to a tensor
    labels = torch.tensor(mapped_labels).long()

    # Debugging: Check the content of labels
    print("Labels before tensor conversion:", labels)


    infer_and_validate_schema(df)
    compute_statistics_and_detect_anomalies(df)
    handle_schema_and_anomalies(df)

    texts = df['text'].tolist()
    mapped_labels = df['mapped_label'].tolist()
    
    # Tokenization for BERT model
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    print("Tokenization completed.")
    print(f"Input IDs shape: {input_ids.shape}")
    attention_mask = inputs["attention_mask"]
    labels = torch.tensor(mapped_labels).long()
    
    # Creating a TensorDataset
    dataset = TensorDataset(input_ids, attention_mask, labels)

    # Splitting the dataset into train, validation, and test sets
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_test_size = total_size - train_size
    val_size = int(0.5 * val_test_size)
    test_size = val_test_size - val_size

    train_dataset, val_test_dataset = random_split(dataset, [train_size, val_test_size])
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    # DataLoader creation
    train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"])
    
    print("Dataset Splitting Details:")
    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")


    print(f"Dataset split into {len(train_dataset)} training, {len(val_dataset)} validation, and {len(test_dataset)} test samples.")

    # Ensure all labels are valid
    if not all(label in label2id.values() for label in labels):
        raise ValueError("Found invalid label values")

    print("DataLoaders created. Starting the training process.")
    optimizer = AdamW(model.parameters(), lr=hyperparams["learning_rate"], weight_decay=hyperparams["weight_decay"])

    for epoch in range(hyperparams["epochs"]):
        print(f"Epoch {epoch+1}/{hyperparams['epochs']} started.")
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(train_loader):
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            grad_norm = np.sqrt(sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None))
            mlflow.log_metric(f"grad_norm_epoch_{epoch+1}", grad_norm)
            optimizer.step()
            print(f"Batch {step+1}/{len(train_loader)}, Loss: {loss.item()}")

        avg_train_loss = total_train_loss / len(train_loader)
        mlflow.log_metric(f"avg_train_loss_epoch_{epoch+1}", avg_train_loss)
        print(f"Training for epoch {epoch+1} completed. Avg Loss: {avg_train_loss}")

        print(f"Starting evaluation for epoch {epoch+1}.")
        model.eval()
        predictions, true_labels = [], []

        for batch in val_loader:
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            predictions.extend(batch_predictions)
            true_labels.extend(label_ids)

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

        mlflow.log_metrics({
            f"validation_accuracy_epoch_{epoch+1}": accuracy,
            f"precision_epoch_{epoch+1}": precision,
            f"recall_epoch_{epoch+1}": recall,
            f"f1_score_epoch_{epoch+1}": f1
        })

        print(f"Evaluation for epoch {epoch+1} completed. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
        
    
    # Confusion Matrix and Classification Report
    conf_matrix = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions, target_names=id2label.values())
    print(f"Confusion Matrix for epoch {epoch+1}:\n{conf_matrix}")
    print(f"Classification Report for epoch {epoch+1}:\n{class_report}")

    # Plotting Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=id2label.values(), yticklabels=id2label.values())
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for Epoch {epoch+1}')
    plt.show()
    
    # Prepare misclassified examples
    errors = [(texts[i], id2label[predictions[i]], id2label[true_labels[i]]) for i in range(len(true_labels)) if predictions[i] != true_labels[i]]
    error_text = "\n".join([f"Text: {e[0]}, Predicted: {e[1]}, Actual: {e[2]}" for e in errors[:100]])

    # Write to a text file
    error_file_path = f"error_analysis_epoch_{epoch+1}.txt"
    with open(error_file_path, "w") as file:
        file.write(error_text)

    # Log the text file
    mlflow.log_artifact(error_file_path)

    # Optionally, remove the file after logging
    os.remove(error_file_path)

    # Log errors and reports
    error_analysis_file = f"error_analysis_epoch_{epoch+1}.txt"
    with open(error_analysis_file, "w") as file:
        for e in errors[:100]:
            file.write(f"Text: {e[0]}, Predicted: {e[1]}, Actual: {e[2]}\n")

    mlflow.log_artifact(error_analysis_file)
    os.remove(error_analysis_file)
    mlflow.log_text(f"classification_report_epoch_{epoch+1}.txt", class_report)
    mlflow.log_artifact(CSV_FILE_PATH, "dataset")
    mlflow.pytorch.log_model(model, "model")

    print(f"Model training and evaluation completed. Model has been logged with run id: {run.info.run_id}")
    print(f"Final Model State Dict Keys: {model.state_dict().keys()}")
    print(f"MLflow run completed with run id: {run.info.run_id}")
    print("Stance Detection Model Training Finished Successfully.")
    # Assuming model is saved as 'model.pth' and dataset is 'dataset.csv'
    torch.save(model.state_dict(), "model.pth")
    df.to_csv("modified_dataset.csv", index=False)
    commit_changes("Add trained model and dataset")
    # Push to remote repository
    push_to_remote("origin", "main")

    print("Git versioning completed.")
