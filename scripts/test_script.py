import pytest
import requests

# Base URL of the running FastAPI application
BASE_URL = "http://localhost:8000"

def test_health_check():
    response = requests.get(f"{BASE_URL}/health_check/")
    assert response.status_code == 200
    assert response.json() == {"status": "running", "message": "API is healthy"}

def test_prediction():
    response = requests.post(
        f"{BASE_URL}/predict/",
        data={"text": "This is a sample text for prediction"}
    )
    assert response.status_code == 200
    result = response.json()
    assert "text" in result
    assert "class_id" in result
    assert "description" in result
    assert result["text"] == "This is a sample text for prediction"

def test_upload_data():
    files = {'file': open('C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\dataset_reduced.csv', 'rb')}
    response = requests.post(f"{BASE_URL}/load_data/", files=files)
    assert response.status_code == 200
    result = response.json()
    assert "message" in result
    assert result["message"] == "Data loaded successfully"

def test_analyze_data():
    files = {'file': open('C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\dataset_reduced.csv', 'rb')}
    response = requests.post(f"{BASE_URL}/analyze_data/", files=files)
    assert response.status_code == 200
    result = response.json()
    assert "message" in result
    assert result["message"] == "Data analysis completed"
    assert "analysis" in result

def test_visualize_data():
    files = {'file': open('C:\\Users\\LENOVO\\Desktop\\CSC5382_SP24_FINALPROJECT\\scripts\\dataset_reduced.csv', 'rb')}
    response = requests.post(f"{BASE_URL}/visualize_data/", files=files)
    assert response.status_code == 200
    result = response.json()
    assert "message" in result
    assert result["message"] == "Data visualization created"
    assert "url" in result

if __name__ == "__main__":
    pytest.main()
