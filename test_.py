from transformers import AutoModel, AutoConfig, AutoTokenizer
import mlflow.pyfunc

# Load the configuration, tokenizer, and model
config = AutoConfig.from_pretrained('/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden/config.json')
tokenizer = AutoTokenizer.from_pretrained('/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden/')
model = AutoModel.from_pretrained('/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden/pytorch_model.bin', config=config)

# A custom wrapper for the MLflow to handle the Hugging Face model
class ModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, context, model_input):
        # Here you would add the code to make predictions using the model and tokenizer
        # For example:
        inputs = self.tokenizer(model_input, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

# Log the model
with mlflow.start_run():
    mlflow.pyfunc.log_model(artifact_path="model", python_model=ModelWrapper())
