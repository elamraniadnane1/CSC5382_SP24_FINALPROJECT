import pandas as pd
from sklearn.utils import resample

def oversample_minority_class(dataframe, target_column):
    """
    Perform oversampling on the minority class.

    :param dataframe: DataFrame containing features and target.
    :param target_column: Name of the target column.
    :return: Balanced DataFrame after oversampling.
    """
    # Separate majority and minority classes
    df_majority = dataframe[dataframe[target_column] == dataframe[target_column].mode()[0]]
    df_minority = dataframe[dataframe[target_column] != dataframe[target_column].mode()[0]]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority, 
                                     replace=True,     
                                     n_samples=len(df_majority),    
                                     random_state=123) 

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    return df_upsampled

def undersample_majority_class(dataframe, target_column):
    """
    Perform undersampling on the majority class.

    :param dataframe: DataFrame containing features and target.
    :param target_column: Name of the target column.
    :return: Balanced DataFrame after undersampling.
    """
    # Separate majority and minority classes
    df_majority = dataframe[dataframe[target_column] == dataframe[target_column].mode()[0]]
    df_minority = dataframe[dataframe[target_column] != dataframe[target_column].mode()[0]]

    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                       replace=False,    
                                       n_samples=len(df_minority),     
                                       random_state=123) 

    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    return df_downsampled

if __name__ == "__main__":
    # Example usage
    file_path = 'C:\\Users\\DELL\\CSC5356_SP24\\scripts\\files\\your_dataset.csv'
    df = pd.read_csv(file_path)

    target_col = 'your_target_column'

    # Apply oversampling
    balanced_df_oversampling = oversample_minority_class(df, target_col)

    # Apply undersampling
    balanced_df_undersampling = undersample_majority_class(df, target_col)

    # Now you can continue with your ML tasks using the balanced datasets
