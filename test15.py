import mlflow
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class BertPyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, tokenizer, label_map):
        self.model = model
        self.tokenizer = tokenizer
        self.label_map = label_map

    def predict(self, context, model_input):
        inputs = self.tokenizer(model_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        labels = torch.argmax(probabilities, dim=1)
        scores, label_indices = torch.topk(probabilities, 1)
        return [(self.label_map[str(label.item())], score.item()) for label, score in zip(label_indices, scores)]

# Define the repo_id of the model on the Hugging Face model hub
MODEL_REPO_ID = "kornosk/bert-election2020-twitter-stance-biden"

# Define the label map
LABEL_MAP = {
    'LABEL_0': 'Neutral Stance',
    'LABEL_1': 'Pro-Biden Stance',
    'LABEL_2': 'Anti-Biden Stance',
}

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(MODEL_REPO_ID)
tokenizer = BertTokenizer.from_pretrained(MODEL_REPO_ID)

# Save the tokenizer to a directory
tokenizer_dir = "tokenizer_dir"
tokenizer.save_pretrained(tokenizer_dir)

# Sample input for signature inference
sample_input = ["Sample text for input"]
sample_output = BertPyfuncModel(model, tokenizer, LABEL_MAP).predict(None, sample_input)

# Set the MLflow tracking URI and start an experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.start_run()

# Define the model path for MLflow
model_path = "bert_model"

# Log the model with MLflow
mlflow.pyfunc.log_model(
    artifact_path=model_path,
    python_model=BertPyfuncModel(model=model, tokenizer=tokenizer, label_map=LABEL_MAP),
    artifacts={"tokenizer": tokenizer_dir},
    signature=infer_signature(sample_input, sample_output),
    conda_env={
        'name': 'mlflow-env',
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
            'python=3.8',
            'pytorch',
            'transformers',
            'mlflow',
            'torch'
        ]
    }
)

# End the MLflow run
mlflow.end_run()
