import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def custom_feature_extractor(df, text_column):
    """
    Extract custom features from the text column of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with text data.
        text_column (str): Name of the column containing text data.

    Returns:
        pd.DataFrame: DataFrame with extracted features.
    """
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Define a function to lemmatize text
    def lemmatize_text(text):
        return ' '.join([lemmatizer.lemmatize(w) for w in text.split()])

    # Lemmatize the text column
    df['lemmatized_text'] = df[text_column].apply(lemmatize_text)

    # Initialize CountVectorizer for some basic text features
    vectorizer = CountVectorizer(max_features=100)  # Adjust the number of features

    # Fit and transform the lemmatized text
    feature_matrix = vectorizer.fit_transform(df['lemmatized_text'])

    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Add more custom features if needed
    # For example, word count, presence of specific words, etc.

    return feature_df

def main():
    file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\dataset.csv'
    output_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\extracted_features.csv'
    
    # Load data
    data = load_data(file_path)

    # Assume 'text' is the column with textual data
    features = custom_feature_extractor(data, 'text')

    # Save extracted features
    features.to_csv(output_file_path, index=False)
    print(f"Extracted features saved to {output_file_path}")

if __name__ == "__main__":
    main()
