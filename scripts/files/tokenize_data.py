# File: tokenize_data.py

from transformers import AutoTokenizer
import logging

def tokenize_data(data, tokenizer_path):
    """
    Tokenizes the text data using a specified tokenizer.

    Args:
        data (dict): A dictionary containing the text data to tokenize. 
                     The text data should be under the key 'text'.
        tokenizer_path (str): The path to the tokenizer.

    Returns:
        dict: A dictionary with the tokenized data.
    """
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Extract text data
        texts = data['text']

        # Tokenize text
        tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        # Update the original data dictionary with tokenized data
        data.update(tokenized)

        return data

    except Exception as e:
        logging.error(f"An error occurred during tokenization: {e}")
        raise
