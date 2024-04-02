import json
import pandas as pd

class SchemaRevisionTool:
    def __init__(self, schema_file):
        self.schema_file = schema_file
        self.load_schema()

    def load_schema(self):
        """
        Load the current schema from a file.
        """
        try:
            with open(self.schema_file, 'r') as file:
                self.schema = json.load(file)
        except FileNotFoundError:
            self.schema = {}

    def save_schema(self):
        """
        Save the current schema to a file.
        """
        with open(self.schema_file, 'w') as file:
            json.dump(self.schema, file, indent=4)

    def update_schema(self, new_schema):
        """
        Update the schema with new definitions.

        :param new_schema: A dictionary representing the new schema.
        """
        self.schema.update(new_schema)
        self.save_schema()

    def validate_data(self, data):
        """
        Validate a DataFrame against the current schema.

        :param data: DataFrame to validate.
        :return: Boolean indicating if data is valid.
        """
        if not set(self.schema.keys()).issubset(data.columns):
            return False
        return all(data[col].dtype.name == self.schema[col] for col in self.schema)

# Example usage
if __name__ == "__main__":
    # Initialize the tool with a schema file
    tool = SchemaRevisionTool('schema.json')

    # Update schema
    new_schema = {
        'Name': 'object',
        'Age': 'int64',
        'Email': 'object'
    }
    tool.update_schema(new_schema)

    # Create sample data and validate
    sample_data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30], 'Email': ['alice@example.com', 'bob@example.com']}
    df = pd.DataFrame(sample_data)
    is_valid = tool.validate_data(df)
    print("Is data valid according to schema?", is_valid)
