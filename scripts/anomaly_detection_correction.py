import tensorflow_data_validation as tfdv
import pandas as pd


# Paths
CSV_FILE_PATH = '/home/dino/Desktop/SP24/scripts/biden_stance_public_3_labeled.csv'
EVAL_CSV_FILE_PATH = '/home/dino/Desktop/SP24/scripts/biden_stance_test_public.csv'

# Load the training and evaluation datasets
train_df = pd.read_csv(CSV_FILE_PATH)
eval_df = pd.read_csv(EVAL_CSV_FILE_PATH)

train_stats = tfdv.generate_statistics_from_dataframe(train_df)
eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

# Visualize the statistics of the training dataset
print("Training Dataset Statistics:")
tfdv.visualize_statistics(train_stats)

# Visualize the statistics of the evaluation dataset
print("Evaluation Dataset Statistics:")
tfdv.visualize_statistics(eval_stats)

anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)
tfdv.display_anomalies(anomalies)
tfdv.display_anomalies(anomalies)

train_stats = tfdv.generate_statistics_from_dataframe(train_df)
eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

schema = tfdv.infer_schema(train_stats)

train_anomalies = tfdv.validate_statistics(statistics=train_stats, schema=schema)

eval_anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)

# Display any anomalies found in the training dataset
print("Anomalies in Training Dataset:")
tfdv.display_anomalies(train_anomalies)

# Display any anomalies found in the evaluation dataset
print("Anomalies in Evaluation Dataset:")
tfdv.display_anomalies(eval_anomalies)


# If you want to overwrite the original file, you can use:
eval_df.to_csv(EVAL_CSV_FILE_PATH, index=False)

# Define a path to save the schema
SCHEMA_FILE = '/home/dino/Desktop/SP24/scripts/schema.txt'

# Write the schema to a file
tfdv.write_schema_text(schema, SCHEMA_FILE)

import pkg_resources
import importlib
importlib.reload(pkg_resources)

import os
import tempfile

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

from tfx_bsl.public import tfxio

import pandas as pd
import re

# Function to clean the text data
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove the '#' from hashtags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces, tabs, and newlines
    return text

# Clean the 'text' column
df['text'] = df['text'].apply(clean_text)

#Overwrite the original file with the cleaned data
df.to_csv(CSV_FILE_PATH, index=False)

