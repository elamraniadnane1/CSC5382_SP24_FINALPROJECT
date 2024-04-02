import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

def train_model():
    """
    Train a simple RandomForestClassifier on the Iris dataset.
    
    :return: Trained model.
    """
    iris = load_iris()
    model = RandomForestClassifier()
    model.fit(iris.data, iris.target)
    return model

def save_model(model, filename):
    """
    Save the trained model to a file.

    :param model: Trained machine learning model.
    :param filename: File path to save the model.
    """
    joblib.dump(model, filename)

def load_model(filename):
    """
    Load a saved model from a file.

    :param filename: File path of the saved model.
    :return: Loaded model.
    """
    return joblib.load(filename)

if __name__ == "__main__":
    # Train the model
    model = train_model()

    # Save the model
    save_model(model, 'random_forest_iris_model.pkl')

    # Optionally, load the model
    loaded_model = load_model('random_forest_iris_model.pkl')
    print("Model loaded successfully:", loaded_model)
