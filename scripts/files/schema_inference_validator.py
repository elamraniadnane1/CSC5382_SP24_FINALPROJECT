import pandas as pd

class SchemaInferenceValidator:
    def __init__(self):
        self.inferred_schema = None

    def infer_schema(self, data):
        """
        Infer schema from a given DataFrame.

        :param data: DataFrame to infer the schema from.
        """
        self.inferred_schema = data.dtypes.apply(lambda x: x.name).to_dict()
        print("Inferred Schema:", self.inferred_schema)

    def validate_data(self, data):
        """
        Validate a DataFrame against the inferred schema.

        :param data: DataFrame to validate.
        :return: Boolean indicating if data conforms to the inferred schema.
        """
        if self.inferred_schema is None:
            raise ValueError("Schema has not been inferred. Please infer the schema first.")

        current_schema = data.dtypes.apply(lambda x: x.name).to_dict()
        return self.inferred_schema == current_schema

# Example usage
if __name__ == "__main__":
    schema_validator = SchemaInferenceValidator()

    # Infer schema from initial dataset
    initial_data = pd.DataFrame({
        'Name': ['Alice', 'Bob'],
        'Age': [25, 30],
        'Email': ['alice@example.com', 'bob@example.com']
    })
    schema_validator.infer_schema(initial_data)

    # Validate another dataset against inferred schema
    new_data = pd.DataFrame({
        'Name': ['Charlie', 'David'],
        'Age': [35, 40],
        'Email': ['charlie@example.com', 'david@example.com']
    })
    is_valid = schema_validator.validate_data(new_data)
    print("Does new data conform to the inferred schema?", is_valid)
