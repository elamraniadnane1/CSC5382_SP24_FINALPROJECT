import random
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

# Load data
def load_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

# Preprocess data
def preprocess_data(data):
    from sklearn.preprocessing import LabelEncoder
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\S+|#\S+', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        return text.lower().strip()
    data['text'] = data['text'].apply(clean_text)
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    return data, label_encoder

# Encode data
def encode_data(data, tokenizer, max_length):
    encoded_data = tokenizer.batch_encode_plus(
        data.text.values,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=max_length,
        return_tensors='pt'
    )
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = torch.tensor(data.label.values)
    return input_ids, attention_masks, labels

# Create DataLoader
def create_dataloader(input_ids, attention_masks, labels, batch_size):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)
    return dataloader

# Define the model training function
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Training Loss: {avg_train_loss}")

    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs.logits
            predictions.append(logits.detach().cpu().numpy())
            true_labels.append(b_labels.detach().cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    return accuracy_score(true_labels, np.argmax(predictions, axis=1))

# Define the search space
param_dist = {
    "batch_size": [8, 16, 32],
    "learning_rate": [2e-5, 3e-5, 5e-5],
    "weight_decay": [0.0, 0.01, 0.1],
    "epochs": [2, 3, 4],
    "max_grad_norm": [1.0, 1.5, 2.0],
    "lr_step_size": [1, 2, 3],
    "lr_gamma": [0.1, 0.5, 0.9]
}

class HyperparameterSearch:
    def __init__(self, model_class, tokenizer_class, model_name, param_dist, data_path, max_length=128):
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.model_name = model_name
        self.param_dist = param_dist
        self.data_path = data_path
        self.max_length = max_length

    def objective(self, params):
        batch_size = int(params['batch_size'])
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        epochs = int(params['epochs'])
        max_grad_norm = params['max_grad_norm']
        lr_step_size = int(params['lr_step_size'])
        lr_gamma = params['lr_gamma']

        data = load_data(self.data_path)
        data, label_encoder = preprocess_data(data)
        train_data, val_data = train_test_split(data, test_size=0.1)
        tokenizer = self.tokenizer_class.from_pretrained(self.model_name)
        input_ids_train, attention_masks_train, labels_train = encode_data(train_data, tokenizer, self.max_length)
        input_ids_val, attention_masks_val, labels_val = encode_data(val_data, tokenizer, self.max_length)
        train_dataloader = create_dataloader(input_ids_train, attention_masks_train, labels_train, batch_size)
        val_dataloader = create_dataloader(input_ids_val, attention_masks_val, labels_val, batch_size)

        model = self.model_class.from_pretrained(self.model_name, num_labels=len(label_encoder.classes_))
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        accuracy = train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, device)
        return -accuracy

    def search(self, n_iter=10):
        results = []
        for _ in range(n_iter):
            params = {k: random.choice(v) for k, v in self.param_dist.items()}
            accuracy = self.objective(params)
            results.append((accuracy, params))
        results.sort(key=lambda x: x[0])
        return results

if __name__ == "__main__":
    data_path = "C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\dataset_reduced.csv"
    searcher = HyperparameterSearch(BertForSequenceClassification, BertTokenizer, "C:\\Users\\LENOVO\\CSC5382_SP24_FINALPROJECT\\scripts\\bert-election2024-twitter-stance-biden", param_dist, data_path)
    best_params = searcher.search(n_iter=20)
    print("Best hyperparameters found:")
    print(best_params[0])
