
# Constants and Hyperparameters
CSV_FILE_PATH = '/home/dino/Desktop/SP24/scripts/biden_stance_public.csv'
PRETRAINED_LM_PATH = '/home/dino/Desktop/SP24/bert-election2020-twitter-stance-biden'
HYPERPARAMS = {
    "batch_size": 8,
    "learning_rate": 2e-5,
    "epochs": 2,
    "weight_decay": 0.01
}
print("Hyperparameters:", HYPERPARAMS)

# Define ZenML steps



# 7. Pipeline Definition
@pipeline

stance_pipeline.run()


