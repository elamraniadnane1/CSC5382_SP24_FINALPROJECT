from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class TFIDFVectorizer:
    def __init__(self):
        """
        Initialize the TFIDF Vectorizer.
        """
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, documents):
        """
        Fit the TFIDF model and transform the documents.

        :param documents: List of document strings.
        :return: TFIDF feature matrix.
        """
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        return tfidf_matrix

    def get_feature_names(self):
        """
        Get the feature names (vocabulary) of the TFIDF model.

        :return: List of feature names.
        """
        return self.vectorizer.get_feature_names()

# Example usage
if __name__ == "__main__":
    # Sample documents
    documents = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?'
    ]

    # Initialize and transform the documents
    tfidf_vectorizer = TFIDFVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Convert to DataFrame for better readability
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names())
    
    print(df)
