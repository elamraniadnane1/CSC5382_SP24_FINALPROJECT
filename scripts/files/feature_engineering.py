import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_features(df):
    """
    Preprocess features through scaling, encoding, and imputation.

    Args:
        df (pd.DataFrame): Data to preprocess.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    # Identify numerical and categorical columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Create transformers for numerical and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Apply transformations to the data
    df_processed = preprocessor.fit_transform(df)
    return pd.DataFrame(df_processed)

def main():
    file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\dataset.csv'
    data = load_data(file_path)

    # Apply feature engineering
    processed_data = preprocess_features(data)
    print("Processed data:")
    print(processed_data.head())

if __name__ == "__main__":
    main()
