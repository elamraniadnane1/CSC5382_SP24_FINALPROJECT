import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the data: handle missing values, encode categorical variables, and scale features.

    Args:
        df (pd.DataFrame): DataFrame to preprocess.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Define numerical and categorical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Create pipelines for different types of features
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ])

    # Apply transformations
    df_processed = preprocessor.fit_transform(df)
    return df_processed

def main():
    input_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\data.csv'
    output_file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\preprocessed_data.csv'

    # Load data
    data = load_data(input_file_path)

    # Preprocess data
    preprocessed_data = preprocess_data(data)

    # Convert the array back to a DataFrame (optional, depending on your needs)
    preprocessed_data_df = pd.DataFrame(preprocessed_data, columns=preprocessed_data.feature_names_out_)

    # Save preprocessed data
    preprocessed_data_df.to_csv(output_file_path, index=False)
    print(f"Preprocessed data saved to {output_file_path}")

if __name__ == "__main__":
    main()
