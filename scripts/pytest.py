import pandas as pd
import pytest
from your_pipeline_module import load_data  # Replace with the actual import

# Sample data to mock CSV file contents
sample_data = {
    "column1": [1, 2, 3],
    "column2": ["a", "b", "c"]
}
sample_df = pd.DataFrame(sample_data)

def mock_read_csv_success(path):
    """Mock function to simulate successful reading of CSV."""
    return sample_df

def mock_read_csv_file_not_found_error(path):
    """Mock function to simulate FileNotFoundError."""
    raise FileNotFoundError(f"File not found: {path}")

def mock_read_csv_parser_error(path):
    """Mock function to simulate pandas.errors.ParserError."""
    raise pd.errors.ParserError(f"Error parsing the file: {path}")

@pytest.fixture
def mock_read_csv(monkeypatch):
    """Fixture to replace pd.read_csv with a mock."""
    monkeypatch.setattr(pd, 'read_csv', mock_read_csv_success)

def test_load_data_success(mock_read_csv):
    """Test load_data with a successful CSV read."""
    result = load_data("path/to/mock/file.csv")
    pd.testing.assert_frame_equal(result, sample_df)

def test_load_data_file_not_found():
    """Test load_data handling FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_data("path/to/nonexistent/file.csv")

def test_load_data_parser_error():
    """Test load_data handling pandas.errors.ParserError."""
    with pytest.raises(pd.errors.ParserError):
        load_data("path/to/incorrectly/formatted/file.csv")
import pandas as pd
import pytest
from your_pipeline_module import preprocess_data  # Replace with actual import

# Sample data to test text preprocessing
sample_data = {
    "text": ["Test @user #hashtag http://example.com", "Another test!"],
    "label": ["positive", "negative"]
}
sample_df = pd.DataFrame(sample_data)

def test_preprocess_data_text_cleaning():
    """
    Test if preprocess_data correctly cleans the text data.
    """
    expected_data = {
        "text": ["test", "another test"],
        "label": ["positive", "negative"]
    }
    expected_df = pd.DataFrame(expected_data)

    result_df = preprocess_data(sample_df)
    assert all(result_df["text"] == expected_df["text"]), "Text cleaning failed"

def test_preprocess_data_missing_columns():
    """
    Test if preprocess_data raises an error for missing expected columns.
    """
    incomplete_data = {"text": ["Some text"]}
    incomplete_df = pd.DataFrame(incomplete_data)

    with pytest.raises(ValueError):
        preprocess_data(incomplete_df)

def test_preprocess_data_label_encoding():
    """
    Test if preprocess_data correctly encodes labels.
    """
    result_df = preprocess_data(sample_df)
    assert 'label' in result_df.columns, "Label column missing after preprocessing"
    assert result_df['label'].dtype != 'object', "Label encoding failed"

# Add more tests if needed for specific cases the same for the other steps

