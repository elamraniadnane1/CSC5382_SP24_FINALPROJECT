import os
import torch
import pandas as pd
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm.notebook import tqdm
import mlflow
import re  # Add this import statement

# Function to load data
def load_data(csv_file_path):
    data = pd.read_csv(csv_file_path)
    data.set_index('tweet_id', inplace=True)
    return data

# Function to preprocess data
def preprocess_data(data):
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\S+|#\S+', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text.lower().strip()
    
    data['text'] = data['text'].apply(clean_text)
    label_mapping = {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2}
    data['label'] = data['label'].map(label_mapping)
    return data

# Function to split data
def split_data(data):
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data

# Function to encode data
def encode_data(data, tokenizer, max_length):
    encoded_data = tokenizer.batch_encode_plus(
        data.text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoded_data['input_ids'], encoded_data['attention_mask'], torch.tensor(data.label.values)

# Function to create data loader
def create_data_loader(input_ids, attention_masks, labels, batch_size):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)

# Function to train the model
def train_model(model, dataloader_train, dataloader_val, optimizer, scheduler, device, epochs):
    model.to(device)
    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        total_loss_train = 0
        for batch in dataloader_train:
            batch = tuple(b.to(device) for b in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            model.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss_train += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_loss_train / len(dataloader_train)
        
        model.eval()
        total_loss_val = 0
        predictions, true_vals = [], []
        with torch.no_grad():
            for batch in dataloader_val:
                batch = tuple(b.to(device) for b in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
                outputs = model(**inputs)
                loss = outputs.loss
                total_loss_val += loss.item()
                logits = outputs.logits
                predictions.append(logits.detach().cpu().numpy())
                true_vals.append(inputs['labels'].cpu().numpy())
        
        avg_val_loss = total_loss_val / len(dataloader_val)
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
        accuracy = accuracy_score(true_vals, np.argmax(predictions, axis=1))
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model_state.bin')
        
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Training loss: {avg_train_loss}')
        print(f'Validation loss: {avg_val_loss}')
        print(f'Accuracy: {accuracy}')
    
    return model

# Function to evaluate the model
def evaluate_model(model, dataloader_test, device):
    model.eval()
    predictions, true_vals = [], []
    with torch.no_grad():
        for batch in dataloader_test:
            batch = tuple(b.to(device) for b in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.append(logits.detach().cpu().numpy())
            true_vals.append(inputs['labels'].cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    accuracy = accuracy_score(true_vals, np.argmax(predictions, axis=1))
    precision = precision_score(true_vals, np.argmax(predictions, axis=1), average='macro')
    recall = recall_score(true_vals, np.argmax(predictions, axis=1), average='macro')
    f1 = f1_score(true_vals, np.argmax(predictions, axis=1), average='macro')
    
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1)
    
    return accuracy, precision, recall, f1

# Main function for A/B testing
def ab_testing(csv_file_path, pretrained_model_path, hyperparams_a, hyperparams_b):
    data = load_data(csv_file_path)
    data = preprocess_data(data)
    train_data, val_data, test_data = split_data(data)
    
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
    
    input_ids_train, attention_masks_train, labels_train = encode_data(train_data, tokenizer, hyperparams_a['max_length'])
    input_ids_val, attention_masks_val, labels_val = encode_data(val_data, tokenizer, hyperparams_a['max_length'])
    input_ids_test, attention_masks_test, labels_test = encode_data(test_data, tokenizer, hyperparams_a['max_length'])
    
    dataloader_train_a = create_data_loader(input_ids_train, attention_masks_train, labels_train, hyperparams_a['batch_size'])
    dataloader_val_a = create_data_loader(input_ids_val, attention_masks_val, labels_val, hyperparams_a['batch_size'])
    dataloader_test_a = create_data_loader(input_ids_test, attention_masks_test, labels_test, hyperparams_a['batch_size'])
    
    dataloader_train_b = create_data_loader(input_ids_train, attention_masks_train, labels_train, hyperparams_b['batch_size'])
    dataloader_val_b = create_data_loader(input_ids_val, attention_masks_val, labels_val, hyperparams_b['batch_size'])
    dataloader_test_b = create_data_loader(input_ids_test, attention_masks_test, labels_test, hyperparams_b['batch_size'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_a = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=3)
    optimizer_a = AdamW(model_a.parameters(), lr=hyperparams_a['learning_rate'], eps=hyperparams_a['optimizer_eps'])
    scheduler_a = get_linear_schedule_with_warmup(optimizer_a, num_warmup_steps=hyperparams_a['num_warmup_steps'], num_training_steps=len(dataloader_train_a) * hyperparams_a['epochs'])
    
    model_b = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=3)
    optimizer_b = AdamW(model_b.parameters(), lr=hyperparams_b['learning_rate'], eps=hyperparams_b['optimizer_eps'])
    scheduler_b = get_linear_schedule_with_warmup(optimizer_b, num_warmup_steps=hyperparams_b['num_warmup_steps'], num_training_steps=len(dataloader_train_b) * hyperparams_b['epochs'])
    
    print("Training model A with hyperparameters:", hyperparams_a)
    model_a = train_model(model_a, dataloader_train_a, dataloader_val_a, optimizer_a, scheduler_a, device, hyperparams_a['epochs'])
    
    print("Training model B with hyperparameters:", hyperparams_b)
    model_b = train_model(model_b, dataloader_train_b, dataloader_val_b, optimizer_b, scheduler_b, device, hyperparams_b['epochs'])
    
    print("Evaluating model A...")
    accuracy_a, precision_a, recall_a, f1_a = evaluate_model(model_a, dataloader_test_a, device)
    
    print("Evaluating model B...")
    accuracy_b, precision_b, recall_b, f1_b = evaluate_model(model_b, dataloader_test_b, device)
    
    print("A/B Testing Results:")
    print(f"Model A - Accuracy: {accuracy_a}, Precision: {precision_a}, Recall: {recall_a}, F1 Score: {f1_a}")
    print(f"Model B - Accuracy: {accuracy_b}, Precision: {precision_b}, Recall: {recall_b}, F1 Score: {f1_b}")

# Define hyperparameters for A and B models
hyperparams_a = {
    "batch_size": 4,
    "learning_rate": 1e-5,
    "epochs": 10,
    "max_length": 150,
    "num_labels": 3,
    "ignore_mismatched_sizes": True,
    "optimizer_eps": 1e-8,
    "num_warmup_steps": 0
}

hyperparams_b = {
    "batch_size": 8,
    "learning_rate": 2e-5,
    "epochs": 10,
    "max_length": 150,
    "num_labels": 3,
    "ignore_mismatched_sizes": True,
    "optimizer_eps": 1e-8,
    "num_warmup_steps": 0
}

# Run A/B testing
csv_file_path = 'C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\dataset_reduced.csv'
pretrained_model_path = 'C:\\Users\\LENOVO\\Desktop\\bert-election2024-twitter-stance-biden'
ab_testing(csv_file_path, pretrained_model_path, hyperparams_a, hyperparams_b)
