from fastapi import FastAPI, HTTPException, File, UploadFile, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
import os
import logging
from functools import lru_cache
from pydantic import BaseModel


logging.basicConfig(level=logging.INFO)
import nest_asyncio
nest_asyncio.apply()
app = FastAPI()

# Assuming static files are also in the scripts directory or a subdirectory therein
app.mount("/scripts", StaticFiles(directory="C:\\Users\\LENOVO\\Desktop\\CSC5356_SP24\\scripts"), name="scripts")

# Setup for templates
templates = Jinja2Templates(directory="C:\\Users\\LENOVO\\Desktop\\CSC5356_SP24\\scripts")

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "C:\\Users\\LENOVO\\Desktop\\bert-election2020-twitter-stance-biden")

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


@app.get("/upload_form/")
async def upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/load_data/")
async def load_data(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        return {"message": "Data loaded successfully", "data": df.head().to_dict()}
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load data")
    
@app.post("/analyze_data/")
async def analyze_data(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        analysis = {
            "head": df.head().to_dict(),
            "describe": df.describe().to_dict(),
            "correlation": df.corr().to_dict()
        }
        return {"message": "Data analysis completed", "analysis": analysis}
    except Exception as e:
        logging.error(f"Failed to analyze data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze data")


@app.post("/visualize_data/")
async def visualize_data(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        plt.figure(figsize=(10, 5))
        df.hist()
        plt_path = "static/histogram.png"
        plt.savefig(plt_path)
        plt.close()
        return {"message": "Data visualization created", "url": f"/scripts/{plt_path}"}
    except Exception as e:
        logging.error(f"Failed to visualize data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to visualize data")

@app.get("/health_check/")
async def health_check():
    return {"status": "running", "message": "API is healthy"}


@app.post("/dynamic_predict/")
async def dynamic_predict(request: Request):
    data = await request.json()
    text = data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter is required")

    tokenizer, model = load_model()
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

    description = {
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    }.get(predicted_class_id, "Unknown")

    return {"text": text, "class_id": predicted_class_id, "description": description}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
