import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import platform
import sys

def load_model(pretrained_LM_path):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)
    return tokenizer, model

def predict(sentence, tokenizer, model):
    # Prediction
    inputs = tokenizer(sentence.lower(), return_tensors="pt")
    outputs = model(**inputs)
    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
    return predicted_probability

def main():
    # MLflow experiment setup
    mlflow.set_experiment("HuggingFace Model Prediction")
    with mlflow.start_run():
        pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-biden"
        tokenizer, model = load_model(pretrained_LM_path)

        # Log environment details
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("system_platform", platform.platform())

        # Log model details
        mlflow.log_param("model_name", pretrained_LM_path)
        mlflow.pytorch.log_model(model, "model")

        # Sample predictions
        sentences = ["Hello World.", "Go Go Biden!!!", "Biden is the worst."]
        id2label = {0: "AGAINST", 1: "FAVOR", 2: "NONE"}

        for sentence in sentences:
            predicted_probability = predict(sentence, tokenizer, model)
            prediction = id2label[np.argmax(predicted_probability)]
            print(f"Sentence: {sentence}\nPrediction: {prediction}")

            # Clean sentence for metric name
            metric_base_name = ''.join(e for e in sentence if e.isalnum() or e in {' ', '-', '_', '.', '/'})

            # Log each prediction
            mlflow.log_metric(f"{metric_base_name}_against", predicted_probability[0])
            mlflow.log_metric(f"{metric_base_name}_favor", predicted_probability[1])
            mlflow.log_metric(f"{metric_base_name}_neutral", predicted_probability[2])

if __name__ == "__main__":
    main()
