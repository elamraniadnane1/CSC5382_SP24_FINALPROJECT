import pandas as pd
from schema import Schema, And, Use

class SchemaValidator:
    def __init__(self, schema):
        """
        Initialize the Schema Validator with a predefined schema.

        :param schema: A schema to validate DataFrame against.
        """
        self.schema = schema

    def validate(self, df):
        """
        Validate a DataFrame against the predefined schema.

        :param df: DataFrame to validate.
        :return: Tuple (is_valid, errors), where is_valid is a boolean indicating if the DataFrame matches the schema, and errors is a list of issues found.
        """
        try:
            # Validate each row against the schema
            for _, row in df.iterrows():
                self.schema.validate(row.to_dict())
            return True, None
        except Exception as e:
            return False, str(e)

# Define your schema
# Example schema:
# {
#     'Name': And(str, len),
#     'Age': And(Use(int), lambda n: 18 <= n <= 99),
#     'Email': And(str, lambda s: "@" in s)
# }
schema_definition = Schema({
    'Name': And(str, len),
    'Age': And(Use(int), lambda n: 18 <= n <= 99),
    'Email': And(str, lambda s: "@" in s)
})

# Example usage
if __name__ == "__main__":
    validator = SchemaValidator(schema_definition)

    # Create a sample DataFrame
    data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 'Invalid'], 'Email': ['alice@example.com', 'bob@example', 'charlie@example.com']}
    sample_df = pd.DataFrame(data)

    # Validate DataFrame
    is_valid, errors = validator.validate(sample_df)
    print("Is DataFrame valid?", is_valid)
    if errors:
        print("Errors:", errors)
