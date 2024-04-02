import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load dataset from a CSV file.

    :param file_path: Path to the CSV file.
    :return: DataFrame with loaded data.
    """
    return pd.read_csv(file_path)

def perform_grid_search(X_train, y_train):
    """
    Perform hyperparameter tuning using GridSearchCV.

    :param X_train: Training feature data.
    :param y_train: Training target data.
    :return: GridSearchCV object after fitting.
    """
    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # Create a SVM classifier
    svm = SVC()

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(svm, param_grid, refit=True, verbose=3, cv=5)
    grid_search.fit(X_train, y_train)

    return grid_search

def main():
    # Load data
    file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\your_dataset.csv'
    data = load_data(file_path)

    # Split data into features and target
    X = data.drop('target_column', axis=1)
    y = data['target_column']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform hyperparameter tuning
    grid_search = perform_grid_search(X_train, y_train)

    # Print the best parameters and the corresponding score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_}")

    # Evaluate the best model on the test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
