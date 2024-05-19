import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import numpy as np
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
import re
import shap
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer

# Load the model and tokenizer
MODEL_PATH = 'C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\bert-election2024-twitter-stance-biden'
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Load your dataset
CSV_FILE_PATH = 'C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\dataset_reduced.csv'
data = pd.read_csv(CSV_FILE_PATH)

# Map labels to numerical values
label_mapping = {'NONE': 0, 'FAVOR': 1, 'AGAINST': 2}
data['label'] = data['label'].map(label_mapping)

# Binarize the scores: Consider 'FAVOR' (1) as positive, 'NONE' (0) and 'AGAINST' (2) as negative
data['score'] = data['label'].apply(lambda x: 1 if x == 1 else 0)

# Prepare data for Aequitas
aequitas_data = pd.DataFrame()
aequitas_data['score'] = data['score']
aequitas_data['label_value'] = data['score'] # Same as score because we already binarized it

# Add demographic data (for example purposes, let's assume we have 'demographic' column in your dataset)
# In a real-world scenario, you should replace this with actual demographic data
aequitas_data['attribute'] = data['demographic'] if 'demographic' in data.columns else np.random.choice(['group1', 'group2'], len(data))

# Group metric calculation
group = Group()
xtab, _ = group.get_crosstabs(aequitas_data)

# Bias calculation
bias = Bias()
bdf = bias.get_disparity_predefined_groups(xtab, original_df=aequitas_data, ref_groups_dict={'attribute': 'group1'}, alpha=0.05, mask_significance=True)

# Fairness calculation
fairness = Fairness()
fdf = fairness.get_group_value_fairness(bdf)

# Display results
print(fdf)

# Check available metrics in fdf
print(fdf.columns)

# Define the metrics to plot based on the available columns
available_metrics = ['tpr_disparity', 'fnr_disparity']  # Update this list based on the printed columns
groups = fdf['attribute_value'].unique()

# Plot results
fig, ax = plt.subplots(len(available_metrics), 1, figsize=(10, 15))

for i, metric in enumerate(available_metrics):
    for group in groups:
        group_data = fdf[fdf['attribute_value'] == group]
        ax[i].bar(group, group_data[metric].values[0], label=f'{group} {metric}')
    ax[i].set_title(f'{metric}')
    ax[i].set_xlabel('Groups')
    ax[i].set_ylabel('Disparity')
    ax[i].legend()

plt.tight_layout()
plt.show()

# SHAP analysis for model explainability
# Tokenize the data
tokenized_data = data['text'].apply(lambda x: tokenizer.encode(x, add_special_tokens=True))

# Pad sequences to the same length
max_len = max([len(i) for i in tokenized_data])
padded_data = np.array([i + [0]*(max_len-len(i)) for i in tokenized_data])

# Define a prediction function
def predict(inputs):
    inputs = torch.tensor(inputs).to(torch.int64)  # Ensure inputs are in the correct format
    with torch.no_grad():
        outputs = model(inputs)[0]
    return outputs.cpu().numpy()

# Create SHAP explainer using KernelExplainer
background = padded_data[:100]  # Use a subset as the background for the explainer
explainer = shap.KernelExplainer(predict, background)

# Select a subset of data to explain
sample_data = padded_data[:10]

# Get SHAP values
shap_values = explainer.shap_values(sample_data)

# Plot SHAP values for the first prediction
shap.summary_plot(shap_values, features=sample_data, feature_names=tokenizer.convert_ids_to_tokens(range(max_len)))

# LIME analysis for model explainability
# Define a wrapper function to make predictions with the tokenizer
class BertWrapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def predict_proba(self, texts):
        tokenized = [self.tokenizer.encode(t, add_special_tokens=True, max_length=max_len, truncation=True, padding='max_length') for t in texts]
        inputs = torch.tensor(tokenized).to(torch.int64)
        with torch.no_grad():
            outputs = self.model(inputs)[0]
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        return probabilities

# Create a LIME explainer
lime_explainer = LimeTextExplainer(class_names=['NONE', 'FAVOR', 'AGAINST'])

# Select an example for explanation
example_text = data['text'][0]

# Generate LIME explanation
bert_wrapper = BertWrapper(model, tokenizer)
lime_exp = lime_explainer.explain_instance(example_text, bert_wrapper.predict_proba, num_features=10)

# Display the LIME explanation
lime_exp.show_in_notebook(text=example_text)
