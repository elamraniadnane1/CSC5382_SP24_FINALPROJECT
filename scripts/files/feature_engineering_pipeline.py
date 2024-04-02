import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def build_feature_engineering_pipeline(numeric_features, categorical_features):
    """
    Construct a feature engineering pipeline.

    Args:
        numeric_features (list): List of names of numeric features.
        categorical_features (list): List of names of categorical features.

    Returns:
        Pipeline: Feature engineering pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    return preprocessor

def main():
    file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\dataset.csv'
    data = load_data(file_path)

    # Define numeric and categorical features
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object', 'bool']).columns

    # Split the data into training and testing sets
    X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)

    # Build and apply feature engineering pipeline
    pipeline = build_feature_engineering_pipeline(numeric_features, categorical_features)
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    print("Transformed training data shape:", X_train_transformed.shape)
    print("Transformed test data shape:", X_test_transformed.shape)

if __name__ == "__main__":
    main()
