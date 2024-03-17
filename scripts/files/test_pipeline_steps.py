@step
def implement_feature_store_step(data: dict) -> Output(str, step("Implement Feature Store")):
    """
    ZenML step to implement a feature store.

    Args:
        data (dict): Data to be used to implement the feature store.

    Returns:
        str: Path to the implemented feature store.
    """
    feature_store_path = implement_feature_store(data)
    return feature_store_path