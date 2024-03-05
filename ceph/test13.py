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

# Define the model path for MLflow
model_path = "bert_model"

# Save the model with MLflow
mlflow.pyfunc.save_model(
    path=model_path,
    python_model=BertPyfuncModel(model=model, tokenizer=tokenizer, label_map=LABEL_MAP),
    artifacts={"tokenizer": tokenizer_dir},
    signature=infer_signature(model_path),
    conda_env=None
)
