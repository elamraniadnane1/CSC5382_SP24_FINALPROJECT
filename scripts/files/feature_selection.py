import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def select_features(X_train, y_train, k=10):
    """
    Select top k features based on univariate statistical tests.

    Args:
        X_train (pd.DataFrame): Training data with features.
        y_train (pd.Series): Training data with target variable.
        k (int): Number of top features to select.

    Returns:
        pd.DataFrame: DataFrame with selected features.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_new = selector.fit_transform(X_train, y_train)
    selected_features = X_train.columns[selector.get_support(indices=True)].tolist()
    return pd.DataFrame(X_train_new, columns=selected_features)

def main():
    file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\dataset.csv'
    data = load_data(file_path)
    X = data.drop('target', axis=1)
    y = data['target']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature selection
    X_train_selected = select_features(X_train, y_train, k=10)
    print(f'Selected Features:\n{X_train_selected.columns}')

if __name__ == "__main__":
    main()
