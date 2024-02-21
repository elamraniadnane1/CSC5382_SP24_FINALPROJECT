from transformers import pipeline

# Create a pipeline for text classification using the specified model
pipe = pipeline("text-classification", model="kornosk/bert-election2020-twitter-stance-biden")

# Example sentences to classify
sentences = ["I support Biden's policies.", "I do not like the current administration.", "It's a beautiful day!"]

# Classify each sentence
for sentence in sentences:
    result = pipe(sentence)
    print(f"Sentence: '{sentence}'")
    print("Classification:", result)
    print()
