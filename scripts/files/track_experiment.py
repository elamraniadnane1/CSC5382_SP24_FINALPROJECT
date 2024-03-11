# File: track_experiment.py
import mlflow

def track_experiment(params, metrics, model_path, artifacts=None):
    """
    Tracks an experiment using MLflow.

    Args:
        params (dict): Hyperparameters and other configuration details.
        metrics (dict): Evaluation metrics of the model.
        model_path (str): Path to the saved model.
        artifacts (dict, optional): Additional artifacts to log. Default is None.

    Returns:
        str: ID of the tracked experiment.
    """
    with mlflow.start_run() as run:
        # Log parameters
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)

        # Log the model
        mlflow.pytorch.log_model(model_path, "model")

        # Log additional artifacts if any
        if artifacts:
            for key, value in artifacts.items():
                mlflow.log_artifact(value, key)

        return run.info.run_id
