import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import shap
import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
import numpy as np
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
import re

# Load the model and tokenizer
MODEL_PATH = 'C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\bert-election2024-twitter-stance-biden'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Load your dataset
CSV_FILE_PATH = 'C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\dataset_reduced.csv'
data = pd.read_csv(CSV_FILE_PATH)

# Preprocess data
def preprocess_data(data):
    def clean_text(text):
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\S+|#\S+', '', text)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower().strip()
        return text
    
    data['text'] = data['text'].apply(clean_text)
    label_mapping = {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2}
    data['label'] = data['label'].map(label_mapping)
    return data

data = preprocess_data(data)

# Adjust sample size based on the dataset size
sample_size = min(100, len(data))
subset_data = data.sample(n=sample_size, random_state=42)
texts = subset_data['text'].values
labels = subset_data['label'].values

# Tokenize the texts
encoded_inputs = tokenizer.batch_encode_plus(
    texts,
    add_special_tokens=True,
    max_length=128,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# Function to get model predictions
def predict(inputs):
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits

inputs = {
    'input_ids': encoded_inputs['input_ids'],
    'attention_mask': encoded_inputs['attention_mask']
}

logits = predict(inputs)
predictions = torch.argmax(logits, dim=1).numpy()

# Binarize labels and predictions for Aequitas
binarized_labels = (labels > 0).astype(int)  # Binarize: 0 -> 0 (NONE), 1 or 2 -> 1 (FAVOR, AGAINST)
binarized_predictions = (predictions > 0).astype(int)

# Aequitas Bias and Fairness Assessment
group = Group()
bias = Bias()
fairness = Fairness()

# Prepare data for Aequitas
aequitas_df = pd.DataFrame({
    'score': binarized_predictions,
    'label_value': binarized_labels
})

print(aequitas_df)

# Add demographic columns if available (e.g., race, gender)
# aequitas_df['race'] = subset_data['race']
# aequitas_df['gender'] = subset_data['gender']

xtab, _ = group.get_crosstabs(aequitas_df)

# Ensure the reference groups dictionary contains all necessary references
# Fixing the KeyError: 0 by checking if the mode() returns a value or not
ref_groups_dict = {'score': xtab['score'].mode().iloc[0] if not xtab['score'].mode().empty else 0}

print(ref_groups_dict)

# Calculate the actual number of attributes in the input dataframe
actual_number_of_attributes = len(aequitas_df.columns)
print(f"Actual number of attributes in the input dataframe: {actual_number_of_attributes}")

# Check if ref_groups_dict has the necessary keys
if len(ref_groups_dict) < actual_number_of_attributes:
    ref_groups_dict['label_value'] = 0

b = bias.get_disparity_predefined_groups(xtab, original_df=aequitas_df, ref_groups_dict=ref_groups_dict)
f = fairness.get_group_value_fairness(b)

print("Aequitas Bias and Fairness Results:")
print(f)

# Compute SHAP values
class BertModelWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, texts):
        inputs = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        logits = predict(inputs)
        return logits.detach().numpy()

# Add a masker for text data
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(BertModelWrapper(model, tokenizer), masker=masker)
shap_values = explainer(texts)

# Visualize SHAP values
shap.summary_plot(shap_values, texts, class_names=['NONE', 'FAVOR', 'AGAINST'])

# LIME for local interpretability
lime_explainer = LimeTextExplainer(class_names=['NONE', 'FAVOR', 'AGAINST'])

def lime_predict_proba(texts):
    inputs = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    logits = predict(inputs)
    proba = torch.nn.functional.softmax(logits, dim=1).detach().numpy()
    return proba

# Explain a single instance with LIME
if sample_size > 0:
    idx = 0  # Index of the instance to explain
    lime_exp = lime_explainer.explain_instance(texts[idx], lime_predict_proba, num_features=10)
    lime_exp.show_in_notebook(text=True)
else:
    print("Dataset is empty after preprocessing.")
