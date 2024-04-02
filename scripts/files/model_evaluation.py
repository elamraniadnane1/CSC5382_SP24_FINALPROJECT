import joblib
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def load_model(filename):
    """
    Load a saved model from a file.

    :param filename: File path of the saved model.
    :return: Loaded model.
    """
    return joblib.load(filename)

def load_test_data(csv_file_path):
    """
    Load test data from a CSV file.

    :param csv_file_path: Path to the CSV file.
    :return: DataFrame with the test data.
    """
    return pd.read_csv(csv_file_path)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data and print out the classification report and confusion matrix.

    :param model: Trained machine learning model.
    :param X_test: Test data features.
    :param y_test: Test data true labels.
    """
    predictions = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    # Load the saved model
    model = load_model('path_to_your_saved_model.pkl')

    # Load test data
    test_data = load_test_data('path_to_your_test_data.csv')
    X_test = test_data.drop('target_column', axis=1)
    y_test = test_data['target_column']

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
