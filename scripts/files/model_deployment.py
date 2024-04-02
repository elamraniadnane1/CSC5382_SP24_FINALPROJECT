from flask import Flask, request, jsonify
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

app = Flask(__name__)

def train_and_save_model():
    """
    Train a RandomForestClassifier and save the model.
    """
    iris = load_iris()
    model = RandomForestClassifier()
    model.fit(iris.data, iris.target)
    joblib.dump(model, 'iris_model.pkl')
    return model

def load_model(filename):
    """
    Load a saved model from a file.

    :param filename: File path of the saved model.
    :return: Loaded model.
    """
    return joblib.load(filename)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the class for a given input using the trained model.

    :return: JSON object with the prediction and status code.
    """
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    # Train and save the model
    train_and_save_model()

    # Load the trained model
    model = load_model('iris_model.pkl')

    # Start the Flask API server
    app.run(port=5000, debug=True)
