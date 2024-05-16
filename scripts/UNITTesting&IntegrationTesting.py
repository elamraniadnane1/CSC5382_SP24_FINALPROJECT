import pytest
import pandas as pd
import torch
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from final_pipeline import load_data, preprocess_data, split_data, encode_data, create_data_loader, train_model, evaluate_model, calculate_business_metrics 

# Paths and Hyperparameters for testing
CSV_FILE_PATH = 'C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\dataset_reduced.csv'
PRETRAINED_MODEL_PATH = 'C:\\Users\\LENOVO\\Desktop\\bert-election2024-twitter-stance-biden'
HYPERPARAMS = {
    "batch_size": 4,
    "learning_rate": 1e-5,
    "epochs": 10,
    "max_length": 150,
    "num_labels": 3,
    "ignore_mismatched_sizes": True,
    "optimizer_eps": 1e-8,
    "num_warmup_steps": 0
}

@pytest.fixture
def data():
    return load_data(CSV_FILE_PATH)

def test_load_data():
    data = load_data(CSV_FILE_PATH)
    assert isinstance(data, pd.DataFrame)
    assert 'text' in data.columns
    assert 'label' in data.columns

def test_preprocess_data(data):
    preprocessed_data = preprocess_data(data)
    assert 'text' in preprocessed_data.columns
    assert 'label' in preprocessed_data.columns
    assert preprocessed_data['text'].dtype == object
    assert preprocessed_data['label'].dtype == int

def test_split_data(data):
    train_data, val_data, test_data = split_data(data)
    assert len(train_data) > 0
    assert len(val_data) > 0
    assert len(test_data) > 0

def test_encode_data(data):
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    input_ids, attention_masks, labels = encode_data(data, tokenizer, HYPERPARAMS['max_length'])
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_masks, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert input_ids.shape[0] == len(data)
    assert attention_masks.shape[0] == len(data)
    assert labels.shape[0] == len(data)

def test_create_data_loader(data):
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    input_ids, attention_masks, labels = encode_data(data, tokenizer, HYPERPARAMS['max_length'])
    dataloader = create_data_loader(input_ids, attention_masks, labels, HYPERPARAMS['batch_size'])
    assert isinstance(dataloader, torch.utils.data.DataLoader)
    for batch in dataloader:
        assert len(batch) == 3
        assert batch[0].shape[0] <= HYPERPARAMS['batch_size']
        assert batch[1].shape[0] <= HYPERPARAMS['batch_size']
        assert batch[2].shape[0] <= HYPERPARAMS['batch_size']

def test_train_model(data):
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    input_ids_train, attention_masks_train, labels_train = encode_data(data, tokenizer, HYPERPARAMS['max_length'])
    dataloader_train = create_data_loader(input_ids_train, attention_masks_train, labels_train, HYPERPARAMS['batch_size'])
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH, num_labels=3)
    optimizer = AdamW(model.parameters(), lr=HYPERPARAMS['learning_rate'], eps=HYPERPARAMS['optimizer_eps'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=HYPERPARAMS['num_warmup_steps'], num_training_steps=len(dataloader_train) * HYPERPARAMS['epochs'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = train_model(model, dataloader_train, dataloader_train, optimizer, scheduler, device, HYPERPARAMS['epochs'])
    assert isinstance(trained_model, BertForSequenceClassification)

def test_evaluate_model(data):
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_PATH)
    input_ids_test, attention_masks_test, labels_test = encode_data(data, tokenizer, HYPERPARAMS['max_length'])
    dataloader_test = create_data_loader(input_ids_test, attention_masks_test, labels_test, HYPERPARAMS['batch_size'])
    model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH, num_labels=3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accuracy, precision, recall, f1, true_vals, predictions = evaluate_model(model, dataloader_test, device)
    assert isinstance(accuracy, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)

def test_calculate_business_metrics():
    true_labels = [0, 1, 2, 1, 0]
    predictions = [0, 1, 0, 1, 2]
    metrics = calculate_business_metrics(true_labels, predictions)
    assert isinstance(metrics, dict)
    assert 'total_cost' in metrics
    assert 'false_positives' in metrics
    assert 'false_negatives' in metrics
    assert 'performance_metrics' in metrics
    assert metrics['performance_metrics']['accuracy'] >= 0
    assert metrics['performance_metrics']['precision'] >= 0
    assert metrics['performance_metrics']['recall'] >= 0
    assert metrics['performance_metrics']['f1_score'] >= 0

if __name__ == '__main__':
    pytest.main()
