import pandas as pd
import pytest
import logging
from ml_pipeline_zenml import (
    load_data,
    preprocess_data,
    visualize_data,
    split_data_1,
    split_data_2,
    split_data_3,
    validate_data,
    feature_engineering,
    train_model
)
from ml_pipeline_zenml import clean_text
# Sample data for testing
sample_csv_path = 'C:\\Users\\LENOVO\\Desktop\\SP24\\scripts\\dataset.csv'  # You should create a sample CSV for testing
@pytest.fixture
def test_successful_data_load():
    # Arrange
    test_csv_path = 'path_to_valid_csv_file'

    # Act
    result = load_data(test_csv_path)

    # Assert
    assert isinstance(result, pd.DataFrame)

def test_file_not_found():
    # Arrange
    non_existent_file_path = 'path_to_non_existent_file'

    # Act and Assert
    with pytest.raises(FileNotFoundError):
        load_data(non_existent_file_path)

def log_capture():
    class LogCaptureHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records = []

        def emit(self, record):
            self.records.append(record)

    handler = LogCaptureHandler()
    logging.getLogger().addHandler(handler)
    yield handler.records
    logging.getLogger().removeHandler(handler)

def test_file_not_found(log_capture):
    # Arrange
    non_existent_file_path = 'path_to_non_existent_file'

    # Act
    with pytest.raises(FileNotFoundError):
        load_data(non_existent_file_path)

    # Assert
    assert len(log_capture) > 0
    assert "File not found" in log_capture[0].message

def test_parsing_error(log_capture):
    # Arrange
    invalid_csv_path = 'path_to_invalid_csv_file'

    # Act
    with pytest.raises(pd.errors.ParserError):
        load_data(invalid_csv_path)

    # Assert
    assert len(log_capture) > 0
    assert "Error parsing the file" in log_capture[0].message

def test_null_values_in_data(log_capture):
    # Arrange
    csv_with_nulls_path = 'path_to_csv_with_nulls'

    # Act
    result = load_data(csv_with_nulls_path)

    # Assert
    assert "Null values found in the dataset" in [record.message for record in log_capture]
    # Additional assertions for the result

def test_text_cleaning():
    # Test removing URLs
    assert clean_text("Check this link: http://example.com") == "check this link"

    # Test removing usernames and hashtags
    assert clean_text("@user hello #world") == "hello"

    # Test removing special characters and numbers
    assert clean_text("Clean this! #2021") == "clean this"

    # Test converting to lowercase
    assert clean_text("THIS IS UPPER") == "this is upper"

    # Test proper tokenization and stripping
    assert clean_text("  This is a    sentence. ") == "this is a sentence"

    # Test empty string
    assert clean_text("") == ""



def test_missing_columns():
    # Arrange
    data = pd.DataFrame({'text': ['sample text']})  # Only 'text' column, no 'label'

    # Act and Assert
    with pytest.raises(ValueError):
        preprocess_data(data)

def test_drop_missing_labels():
    # Arrange
    data = pd.DataFrame({'text': ['text1', 'text2'], 'label': ['FAVOR', None]})

    # Act
    processed_data = preprocess_data(data)

    # Assert
    assert processed_data.isnull().sum().sum() == 0
    assert len(processed_data) == 1

def test_label_encoding():
    # Arrange
    data = pd.DataFrame({'text': ['text1', 'text2'], 'label': ['FAVOR', 'AGAINST']})

    # Act
    processed_data = preprocess_data(data)

    # Assert
    assert set(processed_data['label'].unique()) == {1, 2}

def test_incorrect_label_values():
    # Arrange
    data = pd.DataFrame({'text': ['text'], 'label': ['UNKNOWN']})

    # Act and Assert
    with pytest.raises(ValueError):
        preprocess_data(data)

def test_complete_preprocessing():
    # Arrange
    data = pd.DataFrame({'text': ['@user http://example.com #hashtag', 'Another tweet!'],'label': ['FAVOR', 'AGAINST']})

    # Act
    processed_data = preprocess_data(data)

def test_missing_values_visualization(capsys):
    # Arrange
    data_with_missing_values = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, 6]})
    data_without_missing_values = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Act
    visualize_data(data_with_missing_values)
    captured = capsys.readouterr()

    # Assert
    assert "No missing values found in the dataset." not in captured.out

    # Repeat for data_without_missing_values
    visualize_data(data_without_missing_values)
    captured = capsys.readouterr()
    assert "No missing values found in the dataset." in captured.out

def test_label_distribution_visualization(capsys):
    # Arrange
    data_with_label = pd.DataFrame({'label': [1, 0, 1], 'B': [4, 5, 6]})
    data_without_label = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Act
    visualize_data(data_with_label)
    captured_with = capsys.readouterr()
    visualize_data(data_without_label)
    captured_without = capsys.readouterr()

    # Assert
    assert "Label column not found, skipping label distribution visualization." in captured_without.out
    assert "Label column not found, skipping label distribution visualization." not in captured_with.out

def test_exception_handling(capsys):
    # Arrange
    broken_data = "this is not a dataframe"

    # Act
    with pytest.raises(Exception):
        visualize_data(broken_data)

    # Assert
    captured = capsys.readouterr()
    assert "Error occurred during" in captured.out

def setup_function():
    # Create a sample dataset
    data = pd.DataFrame({'feature': range(100), 'label': range(100)})
    return data

def test_training_data_split():
    data = setup_function()
    train_data = split_data_1(data)
    
    assert len(train_data) == pytest.approx(len(data) * 0.7, 0.05)  # Allowing a 5% tolerance

def test_validation_data_split():
    data = setup_function()
    val_data = split_data_2(data)
    
    assert len(val_data) == pytest.approx(len(data) * 0.1, 0.05)  # Allowing a 5% tolerance

def test_testing_data_split():
    data = setup_function()
    test_data = split_data_3(data)
    
    assert len(test_data) == pytest.approx(len(data) * 0.2, 0.05)  # Allowing a 5% tolerance

def test_data_integrity():
    data = setup_function()
    train_data = split_data_1(data)
    val_data = split_data_2(data)
    test_data = split_data_3(data)

    combined_data = pd.concat([train_data, val_data, test_data]).sort_values('feature').reset_index(drop=True)
    assert data.equals(combined_data)  # Checking if combined splits cover the entire original dataset without overlap

def setup_function():
    # Create sample datasets
    train_data = pd.DataFrame({'feature': range(50), 'label': range(50)})
    val_data = pd.DataFrame({'feature': range(50, 70), 'label': range(50, 70)})
    test_data = pd.DataFrame({'feature': range(70, 100), 'label': range(70, 100)})
    return train_data, val_data, test_data

@patch('tensorflow_data_validation.generate_statistics_from_dataframe')
@patch('tensorflow_data_validation.infer_schema')
@patch('tensorflow_data_validation.validate_statistics')
def test_statistics_generation(mock_validate_statistics, mock_infer_schema, mock_generate_statistics):
    train_data, val_data, test_data = setup_function()

    # Mock TFDV functions
    mock_generate_statistics.return_value = MagicMock()
    mock_infer_schema.return_value = MagicMock()
    mock_validate_statistics.return_value = MagicMock()

    # Act
    result = validate_data(train_data, val_data, test_data)

    # Assert
    assert mock_generate_statistics.called
    assert mock_infer_schema.called
    assert mock_validate_statistics.called
    assert result.equals(test_data)

@patch('tensorflow_data_validation.generate_statistics_from_dataframe')
@patch('tensorflow_data_validation.infer_schema')
@patch('tensorflow_data_validation.validate_statistics')
def test_anomaly_detection(mock_validate_statistics, mock_infer_schema, mock_generate_statistics, capsys):
    train_data, val_data, test_data = setup_function()
    
    # Configure mock to simulate anomaly detection
    anomaly = MagicMock()
    anomaly.anomaly_info = {"anomaly": "info"}
    mock_validate_statistics.return_value = anomaly

    # Act
    validate_data(train_data, val_data, test_data)

    # Assert
    captured = capsys.readouterr()
    assert "Anomalies found" in captured.out

@patch('your_module.Cluster')
@patch('your_module.tokenizer.encode')
def test_feature_engineering(mock_encode, mock_cluster):
    # Arrange
    data = pd.DataFrame({'text': ['sample text']})
    mock_session = MagicMock()
    mock_cluster.return_value.connect.return_value = mock_session
    mock_encode.return_value = [101, 102]  # Mocked token IDs

    # Act
    result = feature_engineering(data)

    # Assert
    mock_encode.assert_called_with('sample text', add_special_tokens=True)
    assert 'tokens' in result.columns
    assert result.at[0, 'tokens'] == [101, 102]
    mock_session.execute.assert_called()  # Ensure Cassandra execute method was called
    mock_session.shutdown.assert_called()  # Ensure the session is closed
    mock_cluster.return_value.shutdown.assert_called()  # Ensure the cluster is closed

@patch('your_module.Cluster')
@patch('your_module.tokenizer.encode', side_effect=Exception("Tokenization error"))
def test_error_handling(mock_encode, mock_cluster):
    data = pd.DataFrame({'text': ['sample text']})
    mock_session = MagicMock()
    mock_cluster.return_value.connect.return_value = mock_session

    # Act and Assert
    with pytest.raises(Exception):
        feature_engineering(data)
    mock_session.shutdown.assert_called()  # Ensure the session is closed even on error
    mock_cluster.return_value.shutdown.assert_called()  # Ensure the cluster is closed even on error

@patch('your_module.torch')
@patch('your_module.mlflow')
@patch('your_module.AdamW')
@patch('your_module.get_linear_schedule_with_warmup')
def test_train_model(mock_schedule, mock_adamw, mock_mlflow, mock_torch):
    # Arrange
    data = MagicMock()  # Mocked DataFrame
    train_data = MagicMock()  # Mocked train DataFrame
    val_data = MagicMock()  # Mocked validation DataFrame
    hyperparams = {'param1': 'value1'}

    # Mock external functions and attributes
    mock_torch.device.return_value = 'cpu'
    mock_torch.cuda.is_available.return_value = False
    mock_model = MagicMock(spec=BertForSequenceClassification)
    mock_torch.BertForSequenceClassification.from_pretrained.return_value = mock_model
    mock_model.return_value = MagicMock()
    
    # Mock F1 score to be always above 75%
    mock_model.return_value.f1_score_func.return_value = 0.76

    # Act
    trained_model = train_model(data, train_data, val_data, hyperparams)

    # Assert
    assert mock_torch.BertForSequenceClassification.from_pretrained.called
    assert mock_model.train.called
    # Check if MLflow logging is called
    assert mock_mlflow.start_run.called
    assert mock_mlflow.log_metric.called
    # Ensure the model is returned
    assert isinstance(trained_model, BertForSequenceClassification)

def test_error_handling():
    # Arrange
    data = MagicMock()  # Mocked DataFrame
    train_data = MagicMock()  # Mocked train DataFrame
    val_data = MagicMock()  # Mocked validation DataFrame
    hyperparams = {'param1': 'value1'}

    # Act and Assert
    with pytest.raises(Exception):
        train_model(data, train_data, val_data, hyperparams)
