import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_data(csv_file_path):
    """
    Load data from a CSV file.

    :param csv_file_path: Path to the CSV file.
    :return: DataFrame with the loaded data.
    """
    return pd.read_csv(csv_file_path)

def train_model(X_train, y_train):
    """
    Train the machine learning model.

    :param X_train: Training data features.
    :param y_train: Training data labels.
    :return: Trained model.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    :param model: Trained machine learning model.
    :param X_test: Test data features.
    :param y_test: Test data labels.
    """
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    # Load data
    df = load_data('path_to_your_dataset.csv')

    # Split data into features and target
    X = df.drop('target_column', axis=1)
    y = df['target_column']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
