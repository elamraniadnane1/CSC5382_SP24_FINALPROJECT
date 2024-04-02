import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def train_model(X_train, y_train):
    """
    Train a RandomForest classifier on the provided data.
    """
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance.
    """
    predictions = model.predict(X_test)
    return accuracy_score(y_test, predictions)

def update_model(model, feedback_data):
    """
    Update the model with new feedback data.
    """
    X_feedback, y_feedback = feedback_data.drop('target', axis=1), feedback_data['target']
    model.fit(X_feedback, y_feedback)
    return model

def main():
    file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\dataset.csv'
    data = load_data(file_path)

    # Split the dataset
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Initial Model Accuracy: {accuracy}')

    # Simulate receiving new feedback data
    feedback_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\feedback_data.csv'
    feedback_data = load_data(feedback_file_path)

    # Update the model with feedback data
    model = update_model(model, feedback_data)

    # Re-evaluate the model
    updated_accuracy = evaluate_model(model, X_test, y_test)
    print(f'Updated Model Accuracy: {updated_accuracy}')

if __name__ == "__main__":
    main()
