from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import os
import logging
import re
import shap
import lime
from lime.lime_text import LimeTextExplainer
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from udit import UDIT
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Assuming static files are also in the scripts directory or a subdirectory therein
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup for templates
templates = Jinja2Templates(directory="templates")

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\bert-election2024-twitter-stance-biden")
    CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\dataset_reduced.csv")

@lru_cache()
def load_model():
    logging.info("Loading the BERT model from path: %s", Config.MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained(Config.MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(Config.MODEL_PATH)
    return tokenizer, model

@app.get("/predict_form/")
async def predict_form(request: Request):
    return templates.TemplateResponse("prediction_form.html", {"request": request})

@app.post("/predict/")
async def predict(text: str = Form(...)):
    label_descriptions = {
        0: "Negative",
        1: "Positive",
        2: "Neutral"
    }
    tokenizer, model = load_model()
    try:
        encoded_input = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=128,
            return_attention_mask=True, return_tensors='pt', truncation=True
        )
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()

        description = label_descriptions.get(predicted_class_id, "Unknown")

        return {"text": text, "class_id": predicted_class_id, "description": description}
    except Exception as e:
        logging.error(f"Failed to process prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/bias_shap_lime_evaluation/")
async def bias_shap_lime_evaluation():
    tokenizer, model = load_model()

    # Load your dataset
    data = pd.read_csv(Config.CSV_FILE_PATH)

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

    # Select a subset of data for evaluation
    subset_data = data.sample(n=100, random_state=42)
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

    # Bias Evaluation with UDIT and Aequitas
    udit = UDIT(model=model, tokenizer=tokenizer)
    audit_results = udit.audit(data=subset_data, text_col='text', label_col='label', predictions=predictions)

    # Aequitas Bias and Fairness Assessment
    group = Group()
    bias = Bias()
    fairness = Fairness()

    aequitas_df = pd.DataFrame({
        'score': predictions,
        'label_value': labels
    })

    g = group.get_crosstabs(aequitas_df)
    b = bias.get_disparity_predefined_groups(g, original_df=aequitas_df, ref_groups_dict={'score': 'FAVOR'})
    f = fairness.get_group_value_fairness(b)

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

    explainer = shap.Explainer(BertModelWrapper(model, tokenizer))
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
    idx = 0  # Index of the instance to explain
    lime_exp = lime_explainer.explain_instance(texts[idx], lime_predict_proba, num_features=10)
    lime_html = lime_exp.as_html()

    return {
        "audit_results": audit_results,
        "aequitas_results": f.to_dict(),
        "shap_values": shap_values.values.tolist(),
        "lime_explanation": lime_html
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
