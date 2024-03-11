# File: save_model.py

import torch
import os

def save_model(model, save_dir, model_name='model.pth'):
    """
    Saves the trained PyTorch model.

    Args:
        model (torch.nn.Module): The trained model to save.
        save_dir (str): The directory to save the model.
        model_name (str): The filename for the saved model.

    Returns:
        str: The path to the saved model.
    """
    # Create the save directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the full path for the model
    model_path = os.path.join(save_dir, model_name)

    # Save the model
    torch.save(model.state_dict(), model_path)

    return model_path
