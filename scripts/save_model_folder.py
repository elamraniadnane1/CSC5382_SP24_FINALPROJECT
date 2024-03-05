from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def download_and_save_model(pretrained_model_name, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    tokenizer.save_pretrained(save_directory)

    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
    model.save_pretrained(save_directory)

pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-biden"
save_directory = "./models/bert-election2020-twitter-stance-biden"

download_and_save_model(pretrained_LM_path, save_directory)
