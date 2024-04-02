from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')

class StopWordsRemover:
    def __init__(self):
        """
        Initialize the Stop Words Remover.
        """
        self.stop_words = set(stopwords.words('english'))

    def remove_stop_words(self, text):
        """
        Remove stop words from the given text.

        :param text: Input text as a string.
        :return: Text string after removing stop words.
        """
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in self.stop_words]
        return ' '.join(filtered_text)

# Example usage
if __name__ == "__main__":
    remover = StopWordsRemover()
    sample_text = "This is an example sentence demonstrating the removal of stop words."
    result = remover.remove_stop_words(sample_text)
    print(result)
