from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

def tokenize_function(example, tokenizer):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def main():
    # Load dataset
    raw_datasets = load_dataset("glue", "mrpc")

    # Load tokenizer
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # Tokenize the dataset
    tokenized_datasets = raw_datasets.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Training arguments
    training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    trainer.train()

    # Evaluate model (Optional)
    predictions = trainer.predict(tokenized_datasets["validation"])
    print(predictions.metrics)

if __name__ == "__main__":
    main()
