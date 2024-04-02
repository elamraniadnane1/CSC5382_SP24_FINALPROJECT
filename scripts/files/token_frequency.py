import re
from collections import Counter
import logging

class TokenFrequencyCalculator:
    def __init__(self, text):
        """
        Initialize the Token Frequency Calculator with text data.

        :param text: A large string or document.
        """
        self.text = text

    def clean_text(self):
        """
        Clean the text data, removing special characters and making it lower case.
        """
        self.text = re.sub(r'\W+', ' ', self.text.lower())

    def calculate_frequency(self):
        """
        Calculate and return the frequency of each token (word) in the text.
        """
        words = self.text.split()
        self.frequency = Counter(words)
        return self.frequency

    def display_frequencies(self, top_n=10):
        """
        Display the top N most common tokens and their frequencies.

        :param top_n: Number of top frequent tokens to display.
        """
        for token, freq in self.frequency.most_common(top_n):
            print(f'{token}: {freq}')

# Example usage
if __name__ == "__main__":
    # Load your data
    text_data = 'Your text data goes here. Replace this string with your actual text data.'

    # Initialize and calculate token frequency
    tf_calculator = TokenFrequencyCalculator(text_data)
    tf_calculator.clean_text()
    frequencies = tf_calculator.calculate_frequency()

    # Display top 10 frequent tokens
    tf_calculator.display_frequencies(top_n=10)
