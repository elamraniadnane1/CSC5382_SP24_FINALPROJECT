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
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Assuming static files are also in the scripts directory or a subdirectory therein
app.mount("/scripts", StaticFiles(directory="C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts"), name="scripts")

# Setup for templates
templates = Jinja2Templates(directory="C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts")

class Config:
    MODEL_PATH = os.getenv("MODEL_PATH", "C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\bert-election2024-twitter-stance-biden")
    CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\dataset_reduced.csv")

@lru_cache()
def load_model():
    logging.info("Loading the BERT model from path: %s", Config.MODEL_PATH)
    try:
        tokenizer = BertTokenizer.from_pretrained(Config.MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(Config.MODEL_PATH)
        logging.info("Model and tokenizer loaded successfully")
        return tokenizer, model
    except Exception as e:
        logging.error(f"Error loading model or tokenizer: {str(e)}")
        raise

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
    logging.info("Received text for prediction: %s", text)
    
    tokenizer, model = load_model()
    try:
        encoded_input = tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=128,
            return_attention_mask=True, return_tensors='pt', truncation=True
        )
        input_ids = encoded_input['input_ids']
        attention_mask = encoded_input['attention_mask']
        
        logging.info("Input encoded successfully")

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class_id = logits.argmax().item()
            logging.info("Prediction made successfully: class_id=%d", predicted_class_id)

        description = label_descriptions.get(predicted_class_id, "Unknown")

        return {"text": text, "class_id": predicted_class_id, "description": description}
    except ValueError as ve:
        logging.error(f"Value error during prediction: {str(ve)}")
        raise HTTPException(status_code=400, detail="Bad request. Please check the input.")
    except Exception as e:
        logging.error(f"Failed to process prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/upload_form/")
async def upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.post("/load_data/")
async def load_data(file: UploadFile = File(...)):
    try:
        # Validate file extension
        if not file.filename.endswith('.csv'):
            logging.error(f"File {file.filename} is not a CSV file")
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
        
        # Read CSV file
        df = pd.read_csv(file.file)
        logging.info(f"Successfully loaded data from {file.filename}")
        
        # Return the first few rows of the dataframe and its shape
        return {
            "message": "Data loaded successfully",
            "data": df.head().to_dict(),
            "shape": {"rows": df.shape[0], "columns": df.shape[1]}
        }
    except pd.errors.EmptyDataError:
        logging.error(f"File {file.filename} is empty")
        raise HTTPException(status_code=400, detail="The uploaded CSV file is empty.")
    except pd.errors.ParserError:
        logging.error(f"File {file.filename} contains parsing errors")
        raise HTTPException(status_code=400, detail="The uploaded CSV file contains parsing errors.")
    except Exception as e:
        logging.error(f"Failed to load data from {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load data due to an internal server error")



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
    uvicorn.run(app, host="0.0.0.0", port=8000)
