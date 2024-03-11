import mlflow
from mlflow.tracking import MlflowClient

def track_experiment(hyperparams: dict, evaluation_metrics: dict, model_path: str, other_artifacts: dict = None):
    """
    Tracks an experiment using MLflow.

    Args:
        hyperparams (dict): Hyperparameters used for training.
        evaluation_metrics (dict): Evaluation metrics of the model.
        model_path (str): Path to the saved model.
        other_artifacts (dict, optional): Additional artifacts to log.

    Returns:
        str: Experiment ID.
    """

    # Start MLflow run
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_params(hyperparams)

        # Log evaluation metrics
        mlflow.log_metrics(evaluation_metrics)

        # Log model
        mlflow.log_artifact(model_path, "model")

        # Log additional artifacts if provided
        if other_artifacts:
            for key, value in other_artifacts.items():
                mlflow.log_artifact(value, key)

        # Optional: Log additional information like tags, environment, etc.
        # mlflow.set_tag("tag_key", "tag_value")

        return run.info.run_id

# Example usage
if __name__ == "__main__":
    # Example hyperparameters and metrics
    hyperparams = {"learning_rate": 0.01, "batch_size": 32}
    evaluation_metrics = {"accuracy": 0.95, "loss": 0.05}

    # Assuming a model is saved at this path
    model_path = "path/to/your/model.pth"

    # Track the experiment
    experiment_id = track_experiment(hyperparams, evaluation_metrics, model_path)
    print(f"Tracked experiment ID: {experiment_id}")
