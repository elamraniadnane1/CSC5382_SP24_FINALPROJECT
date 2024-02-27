import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

def load_model(pretrained_LM_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)
    return tokenizer, model

def predict(sentence, tokenizer, model):
    inputs = tokenizer(sentence.lower(), return_tensors="pt")
    outputs = model(**inputs)
    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
    return predicted_probability

def main():
    mlflow.set_experiment("HuggingFace Model Prediction")
    with mlflow.start_run():
        pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-biden"
        tokenizer, model = load_model(pretrained_LM_path)
        mlflow.log_param("model_name", pretrained_LM_path)

        # Log the model
        mlflow.pytorch.log_model(model, "model")

        # Register the model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        registered_model_name = "Election2024StancePredictionModel"
        mlflow.register_model(model_uri, registered_model_name)

        # Optional: Transition the model to a stage (e.g., staging, production)
        # mlflow.register_model(model_uri, "Election2024StancePredictionModel")
        # client = mlflow.tracking.MlflowClient()
        # client.transition_model_version_stage(
        #     name=registered_model_name,
        #     version=1,
        #     stage="Staging"
        # )

        sentences = ["Hello World.", "Go Go Biden!!!", "Biden is the worst."]
        id2label = {0: "AGAINST", 1: "FAVOR", 2: "NONE"}

        for sentence in sentences:
            predicted_probability = predict(sentence, tokenizer, model)
            prediction = id2label[np.argmax(predicted_probability)]
            print(f"Sentence: {sentence}\nPrediction: {prediction}")
            metric_base_name = ''.join(e for e in sentence if e.isalnum() or e in {' ', '-', '_', '.', '/'})
            mlflow.log_metric(f"{metric_base_name}_against", predicted_probability[0])
            mlflow.log_metric(f"{metric_base_name}_favor", predicted_probability[1])
            mlflow.log_metric(f"{metric_base_name}_neutral", predicted_probability[2])

if __name__ == "__main__":
    main()

