"""
Unit tests for Crop Recommendation System
Run with: python -m pytest test_app.py
"""

import pytest
import json
from app import app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test home page loads"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Crop Recommendation System' in response.data

def test_valid_prediction_api(client):
    """Test valid API prediction"""
    data = {
        "Nitrogen": 90,
        "Phosphorus": 42,
        "Potassium": 43,
        "Temperature": 20.87,
        "Humidity": 82.00,
        "Ph": 6.50,
        "Rainfall": 202.93
    }
    response = client.post('/api/predict',
                          data=json.dumps(data),
                          content_type='application/json')
    assert response.status_code == 200
    json_data = json.loads(response.data)
    assert 'crop' in json_data
    assert 'confidence' in json_data
    assert json_data['success'] == True

def test_missing_parameter_api(client):
    """Test API with missing parameter"""
    data = {
        "Nitrogen": 90,
        "Phosphorus": 42,
        # Missing Potassium
        "Temperature": 20.87,
        "Humidity": 82.00,
        "Ph": 6.50,
        "Rainfall": 202.93
    }
    response = client.post('/api/predict',
                          data=json.dumps(data),
                          content_type='application/json')
    assert response.status_code == 400

def test_out_of_range_parameter_api(client):
    """Test API with out of range parameter"""
    data = {
        "Nitrogen": 200,  # Out of range (max 140)
        "Phosphorus": 42,
        "Potassium": 43,
        "Temperature": 20.87,
        "Humidity": 82.00,
        "Ph": 6.50,
        "Rainfall": 202.93
    }
    response = client.post('/api/predict',
                          data=json.dumps(data),
                          content_type='application/json')
    assert response.status_code == 400

def test_history_page(client):
    """Test history page loads"""
    response = client.get('/history')
    assert response.status_code == 200
    assert b'Prediction History' in response.data

def test_clear_history(client):
    """Test clear history functionality"""
    response = client.get('/clear_history')
    assert response.status_code == 200

def test_valid_form_submission(client):
    """Test valid form submission"""
    data = {
        "Nitrogen": "90",
        "Phosphorus": "42",
        "Potassium": "43",
        "Temperature": "20.87",
        "Humidity": "82.00",
        "Ph": "6.50",
        "Rainfall": "202.93"
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b'Recommended Crop' in response.data or b'confidence' in response.data.lower()

def test_invalid_form_submission(client):
    """Test invalid form submission"""
    data = {
        "Nitrogen": "invalid",  # Invalid value
        "Phosphorus": "42",
        "Potassium": "43",
        "Temperature": "20.87",
        "Humidity": "82.00",
        "Ph": "6.50",
        "Rainfall": "202.93"
    }
    response = client.post('/predict', data=data)
    assert response.status_code == 200
    assert b'error' in response.data.lower() or b'Error' in response.data

# Sample test data for different crops
SAMPLE_DATA = {
    'rice': {
        "Nitrogen": 90, "Phosphorus": 42, "Potassium": 43,
        "Temperature": 20.87, "Humidity": 82.00, 
        "Ph": 6.50, "Rainfall": 202.93
    },
    'coffee': {
        "Nitrogen": 100, "Phosphorus": 20, "Potassium": 30,
        "Temperature": 25.50, "Humidity": 50.50,
        "Ph": 6.50, "Rainfall": 150.00
    },
    'apple': {
        "Nitrogen": 20, "Phosphorus": 100, "Potassium": 200,
        "Temperature": 22.00, "Humidity": 60.00,
        "Ph": 5.50, "Rainfall": 150.00
    }
}

@pytest.mark.parametrize("crop_data", SAMPLE_DATA.values())
def test_multiple_crop_predictions(client, crop_data):
    """Test predictions for different crop types"""
    response = client.post('/api/predict',
                          data=json.dumps(crop_data),
                          content_type='application/json')
    assert response.status_code == 200
    json_data = json.loads(response.data)
    assert 'crop' in json_data
    assert isinstance(json_data['confidence'], float)
    assert 0 <= json_data['confidence'] <= 100

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
