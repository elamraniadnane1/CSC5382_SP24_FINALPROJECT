from gensim.models import Word2Vec
import logging

# Enable logging for monitoring training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class WordEmbeddingsGenerator:
    def __init__(self, sentences, size=100, window=5, min_count=5, workers=4):
        """
        Initialize the Word Embeddings Generator.

        :param sentences: List of tokenized sentences.
        :param size: The dimensionality of the word vectors.
        :param window: The maximum distance between a target word and words around the target word.
        :param min_count: Ignores all words with total frequency lower than this.
        :param workers: Use these many worker threads to train the model.
        """
        self.sentences = sentences
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def train_model(self):
        """
        Train the Word2Vec model with the given sentences.
        """
        self.model = Word2Vec(self.sentences, size=self.size, window=self.window, min_count=self.min_count, workers=self.workers)
        self.model.train(self.sentences, total_examples=len(self.sentences), epochs=10)

    def save_model(self, file_name='word2vec.model'):
        """
        Save the trained model.

        :param file_name: Name of the file to save the model.
        """
        self.model.save(file_name)

    def load_model(self, file_name='word2vec.model'):
        """
        Load a trained model.

        :param file_name: Name of the file to load the model from.
        """
        self.model = Word2Vec.load(file_name)

    def get_vector(self, word):
        """
        Get the vector for a given word.

        :param word: The word to get the vector for.
        :return: The vector for the given word.
        """
        return self.model.wv[word]

# Example usage
if __name__ == "__main__":
    # Load your data: list of sentences
    # sentences = [['first', 'sentence'], ['second', 'sentence'], ...]
    sentences = ...  # Replace with your data

    # Initialize and train the model
    we_generator = WordEmbeddingsGenerator(sentences)
    we_generator.train_model()

    # Save the model
    we_generator.save_model('my_word_embeddings.model')

    # Load the model (if needed)
    we_generator.load_model('my_word_embeddings.model')

    # Get vector for a word
    vector = we_generator.get_vector('example')
    print(vector)
