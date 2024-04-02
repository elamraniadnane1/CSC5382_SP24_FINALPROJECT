import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

class OrdinalFeatureTransformer:
    def __init__(self, ordinal_mappings):
        """
        Initialize with the specified ordinal mappings.

        :param ordinal_mappings: Dictionary of ordinal feature mappings. 
                                 Each key is a column name, and its value is a list of ordered categories.
        """
        self.ordinal_mappings = ordinal_mappings
        self.encoders = {feature: OrdinalEncoder(categories=[mapping]) 
                         for feature, mapping in ordinal_mappings.items()}

    def fit_transform(self, df):
        """
        Fit the ordinal encoders and transform the DataFrame.

        :param df: DataFrame to be transformed.
        :return: Transformed DataFrame.
        """
        for feature, encoder in self.encoders.items():
            if feature in df.columns:
                transformed = encoder.fit_transform(df[[feature]])
                df[feature] = transformed
        return df

    def inverse_transform(self, df):
        """
        Inverse transform the ordinal features to their original categories.

        :param df: DataFrame to be inverse transformed.
        :return: DataFrame with original categories.
        """
        for feature, encoder in self.encoders.items():
            if feature in df.columns:
                inversed = encoder.inverse_transform(df[[feature]])
                df[feature] = inversed.flatten()
        return df

# Example usage
if __name__ == "__main__":
    # Define ordinal mappings
    ordinal_mappings = {
        'Size': ['Small', 'Medium', 'Large']
    }

    transformer = OrdinalFeatureTransformer(ordinal_mappings)

    # Sample data
    sample_data = {'Size': ['Medium', 'Large', 'Small', 'Small']}
    df = pd.DataFrame(sample_data)

    # Transform data
    transformed_df = transformer.fit_transform(df)
    print("Transformed DataFrame:\n", transformed_df)

    # Inverse transform data
    inversed_df = transformer.inverse_transform(transformed_df)
    print("Inversed DataFrame:\n", inversed_df)
