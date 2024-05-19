from transformers import BertTokenizer, BertForSequenceClassification
import torch
import subprocess
import os

MODEL_PATH = 'C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\bert-election2024-twitter-stance-biden'
SAVE_PATH = 'C:\\Users\\LENOVO\\Desktop\\saved_model'
HANDLER_PATH = 'C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\transformers_handler.py'  # Update this path as needed

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Save the tokenizer and model
tokenizer.save_pretrained(SAVE_PATH)
model.save_pretrained(SAVE_PATH)

# If the model uses PyTorch, save the model as a .bin file
model_file_path = f"{SAVE_PATH}/pytorch_model.bin"
torch.save(model.state_dict(), model_file_path)

# Determine the PyTorch version
torch_version = torch.__version__
print(f"PyTorch version: {torch_version}")

# Prepare the model for TorchServe
# Install torch-model-archiver if not already installed
try:
    import torch_model_archiver
except ImportError:
    subprocess.run(["pip", "install", "torch-model-archiver"])

# Create model_store directory if it doesn't exist
model_store_path = os.path.join(SAVE_PATH, "model_store")
os.makedirs(model_store_path, exist_ok=True)

# Archive the model
archive_command = [
    "torch-model-archiver",
    "--model-name", "bert-election2024",
    "--version", "1.0",
    "--serialized-file", model_file_path,
    "--handler", HANDLER_PATH,
    "--export-path", model_store_path,
    "--extra-files", f"{SAVE_PATH}/config.json,{SAVE_PATH}/vocab.txt",
    "--force"
]
subprocess.run(archive_command)

# Verify model archive
archive_file = os.path.join(model_store_path, "bert-election2024.mar")
if os.path.exists(archive_file):
    print(f"Model archive created at: {archive_file}")
else:
    print("Failed to create model archive.")
