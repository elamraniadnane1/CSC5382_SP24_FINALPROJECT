from textblob import download_corpora
download_corpora.download_all()
from textblob import TextBlob

class SentimentAnalyzer:
    def __init__(self):
        """
        Initialize the Sentiment Analyzer.
        """

    def analyze_sentiment(self, text):
        """
        Analyze the sentiment of the given text.

        :param text: Text to analyze.
        :return: A dictionary containing polarity and subjectivity.
        """
        analysis = TextBlob(text)
        return {
            "polarity": analysis.sentiment.polarity,
            "subjectivity": analysis.sentiment.subjectivity
        }

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample_text = "TextBlob is a great tool for simple NLP tasks."
    sentiment = analyzer.analyze_sentiment(sample_text)
    print(f"Sentiment Analysis: {sentiment}")
