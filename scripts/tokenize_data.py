from zenml.steps import step, Output
from typing import Annotated
from scripts.tokenize_data import tokenize_data

@step
def tokenize_data_step(data: dict) -> Annotated[Output[dict], step("Tokenize Data")]:
    """
    ZenML step to tokenize data.

    Args:
        data (dict): Data to tokenize.

    Returns:
        dict: Tokenized data.
    """
    # Tokenize data using the tokenize_data function from scripts.tokenize_data
    tokenized_data = tokenize_data(data)
    return tokenized_data
