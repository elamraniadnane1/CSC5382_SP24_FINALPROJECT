import mlflow

# Set the experiment name
mlflow.set_experiment("Milestone 2")

# Start an MLflow run
with mlflow.start_run():
    # Your experiment code here
    # For example, log a parameter
    mlflow.log_param("param1", "value1")

    # Log a metric
    test_accuracy = 0.95  # Example metric
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Optionally, log artifacts like models or plots
    # mlflow.log_artifact("path/to/your/artifact")

# The run is automatically closed at the end of the with block
