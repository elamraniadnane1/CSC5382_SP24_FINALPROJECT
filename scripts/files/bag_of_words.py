import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def generate_bag_of_words(df, text_column):
    """
    Generate a Bag of Words representation from a text column in a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        text_column (str): Name of the column containing text data.

    Returns:
        pd.DataFrame: DataFrame with Bag of Words representation.
    """
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(df[text_column])

    # Convert to DataFrame for easier manipulation
    bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    return bow_df

def main():
    file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\text_data.csv'
    output_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\bow_data.csv'

    # Load data
    data = load_data(file_path)

    # Generate Bag of Words
    bow_data = generate_bag_of_words(data, 'text_column')  # Replace 'text_column' with your text column name

    # Save the Bag of Words data
    bow_data.to_csv(output_file_path, index=False)
    print(f"Bag of Words data saved to {output_file_path}")

if __name__ == "__main__":
    main()
