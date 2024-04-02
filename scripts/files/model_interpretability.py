import shap
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def train_and_save_model():
    """
    Train a RandomForestClassifier and save the model.
    """
    iris = load_iris()
    model = RandomForestClassifier()
    model.fit(iris.data, iris.target)
    joblib.dump(model, 'rf_iris_model.pkl')
    return iris, model

def load_model(filename):
    """
    Load a saved model from a file.

    :param filename: File path of the saved model.
    :return: Loaded model.
    """
    return joblib.load(filename)

def explain_model_predictions(model, data):
    """
    Use SHAP to explain model predictions.

    :param model: Trained machine learning model.
    :param data: Data used for training the model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data.data)

    # Plot summary plot using SHAP values
    shap.summary_plot(shap_values, data.data, feature_names=data.feature_names)
    
    # Optionally, return the explainer and SHAP values for further analysis
    return explainer, shap_values

if __name__ == "__main__":
    # Train and save the model
    iris_data, trained_model = train_and_save_model()

    # Load the model
    model = load_model('rf_iris_model.pkl')

    # Explain model predictions
    _, shap_values = explain_model_predictions(model, iris_data)

    # Show plot
    plt.show()
