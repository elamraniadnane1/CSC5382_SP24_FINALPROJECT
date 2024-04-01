

from transformers import AdamW, get_linear_schedule_with_warmup
import mlflow
import torch
from tqdm.notebook import tqdm
from zenml.pipelines import pipeline
from zenml.steps import step
from sklearn.model_selection import train_test_split


HYPERPARAMS_A = {
    "batch_size": 4,             
    "learning_rate": 1e-5,       
    "epochs": 10
}

HYPERPARAMS_B = {
    "batch_size": 4,             
    "learning_rate": 2e-5,       
    "epochs": 10
}
@step
def train_model(data, train_data, val_data, hyperparams, variant_name):
    # Initialize the device for training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize and configure the model based on hyperparameters
    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_LM_PATH,
        num_labels=hyperparams['num_labels'],
        ignore_mismatched_sizes=hyperparams['ignore_mismatched_sizes']
    )
    model.to(device)

    # Prepare data loaders, optimizers, etc.
    train_loader = DataLoader(train_data, batch_size=hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=hyperparams['batch_size'], shuffle=False)
    optimizer = AdamW(model.parameters(), lr=hyperparams['learning_rate'], eps=hyperparams['optimizer_eps'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=hyperparams['num_warmup_steps'], num_training_steps=len(train_loader)*hyperparams['epochs'])

    # Training loop
    for epoch in range(hyperparams['epochs']):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        avg_train_loss = total_loss / len(train_loader)

        # Evaluation on the validation set
        model.eval()
        val_accuracy = 0 # Placeholder for actual accuracy calculation
        # Include your logic for evaluating the model on the validation set and calculating accuracy

        # Log the metrics
        mlflow.log_metric(f"{variant_name}_training_loss", avg_train_loss, step=epoch)
        mlflow.log_metric(f"{variant_name}_validation_accuracy", val_accuracy, step=epoch)

    # Return the model and any relevant metrics
    metrics = {
        'training_loss': avg_train_loss,
        'validation_accuracy': val_accuracy
    }
    return model, metrics


@pipeline
def biden_stance_pipeline_ab_test(
    load_data_step,
    preprocess_data_step,
    train_model_step,
    split_data_step  # Assuming you have a step to split data
):
    # Load and preprocess data
    data = load_data_step()
    preprocessed_data = preprocess_data_step(data=data)

    # Split data
    train_data, val_data, test_data = split_data_step(data=preprocessed_data)

    # Train and evaluate Variant A
    model_a, metrics_a = train_model_step(
        data=preprocessed_data,
        train_data=train_data, 
        val_data=val_data, 
        hyperparams=HYPERPARAMS_A,
        variant_name="A"
    )

    # Train and evaluate Variant B
    model_b, metrics_b = train_model_step(
        data=preprocessed_data,
        train_data=train_data, 
        val_data=val_data, 
        hyperparams=HYPERPARAMS_B,
        variant_name="B"
    )


    # For example:
    print("Comparison of Model A and Model B:")
    print(f"Model A - Accuracy: {metrics_a['validation_accuracy']}, Loss: {metrics_a['training_loss']}")
    print(f"Model B - Accuracy: {metrics_b['validation_accuracy']}, Loss: {metrics_b['training_loss']}")

# Instantiate and run the pipeline
biden_stance_pipeline = biden_stance_pipeline_ab_test(
    load_data_step=load_data(),
    preprocess_data_step=preprocess_data(),
    train_model_step=train_model(),
    split_data_step=split_data()  # Assuming you have this step
)

biden_stance_pipeline.run()
@step
def compare_models(metrics_a, metrics_b, variant_a_name="A", variant_b_name="B"):
    print(f"Comparison between models {variant_a_name} and {variant_b_name}:")

    # Assuming 'accuracy' and 'training_loss' are keys in your metrics dictionaries
    accuracy_a = metrics_a.get('validation_accuracy')
    accuracy_b = metrics_b.get('validation_accuracy')

    loss_a = metrics_a.get('training_loss')
    loss_b = metrics_b.get('training_loss')

    print(f"Model {variant_a_name} - Accuracy: {accuracy_a}, Loss: {loss_a}")
    print(f"Model {variant_b_name} - Accuracy: {accuracy_b}, Loss: {loss_b}")


    # Return some result or decision if needed
    return {
        "better_model": variant_a_name if accuracy_a > accuracy_b else variant_b_name
    }
@pipeline

# Instantiate and run the pipeline
biden_stance_pipeline = biden_stance_pipeline_ab_test(
    load_data_step=load_data(),
    preprocess_data_step=preprocess_data(),
    train_model_step=train_model(),
    split_data_step=split_data(),
    compare_models_step=compare_models()
)

biden_stance_pipeline.run()
@pipeline
def biden_stance_pipeline_ab_test(
    load_data_step,
    preprocess_data_step,
    train_model_step,
    split_data_step,
    compare_models_step
):
    # Load and preprocess data
    data = load_data_step()
    preprocessed_data = preprocess_data_step(data=data)

    # Split your data as required
    train_data, val_data, test_data = split_data_step(data=preprocessed_data)

    # Train and evaluate Variant A
    model_a, metrics_a = train_model_step(
        data=preprocessed_data,
        train_data=train_data, 
        val_data=val_data, 
        hyperparams=HYPERPARAMS_A,
        variant_name="A"
    )

    # Train and evaluate Variant B
    model_b, metrics_b = train_model_step(
        data=preprocessed_data,
        train_data=train_data, 
        val_data=val_data, 
        hyperparams=HYPERPARAMS_B,
        variant_name="B"
    )

    # Compare the two models
    comparison_result = compare_models_step(
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        variant_a_name="A",
        variant_b_name="B"
    )


    better_model = comparison_result['better_model']
    print(f"The better model based on the comparison is: {better_model}")

# Instantiate and run the pipeline
biden_stance_pipeline = biden_stance_pipeline_ab_test(
    load_data_step=load_data(),
    preprocess_data_step=preprocess_data(),
    train_model_step=train_model(),
    split_data_step=split_data(),
    compare_models_step=compare_models()
)

biden_stance_pipeline.run()
