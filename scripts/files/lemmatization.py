import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pandas as pd

# Ensure you have the necessary NLTK data downloaded
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

def get_wordnet_pos(word):
    """
    Map POS tag to first character lemmatize() accepts.

    :param word: The word to be tagged.
    :return: Corresponding tag for lemmatization.
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    """
    Apply lemmatization to a given text.

    :param text: Text to be lemmatized.
    :return: Lemmatized text.
    """
    lemmatizer = WordNetLemmatizer()
    word_list = nltk.word_tokenize(text)
    lemmatized_output = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_list])
    return lemmatized_output

def process_dataframe(df, text_column):
    """
    Apply lemmatization to a specific column in a DataFrame.

    :param df: DataFrame containing text data.
    :param text_column: Column name containing the text to be lemmatized.
    :return: DataFrame with the lemmatized text column.
    """
    df[text_column] = df[text_column].apply(lambda x: lemmatize_text(x))
    return df

if __name__ == '__main__':
    # Example usage with a DataFrame
    df = pd.DataFrame({
        'text': ['He is running quickly', 'Cats eat mice']
    })
    processed_df = process_dataframe(df, 'text')
    print(processed_df)
