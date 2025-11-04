# API Documentation

## Crop Recommendation System API

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Get Prediction (JSON API)
**Endpoint:** `/api/predict`  
**Method:** `POST`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "Nitrogen": 90,
  "Phosphorus": 42,
  "Potassium": 43,
  "Temperature": 20.87,
  "Humidity": 82.00,
  "Ph": 6.50,
  "Rainfall": 202.93
}
```

**Success Response (200):**
```json
{
  "success": true,
  "crop": "Rice",
  "confidence": 95.67,
  "crop_info": {
    "name": "Rice",
    "season": "Kharif (June-October)",
    "water_needs": "High (1200-1800mm)",
    "soil_type": "Clayey loam",
    "growing_period": "3-6 months",
    "temperature": "20-35°C",
    "description": "Rice is a staple food crop...",
    "fertilizer": "NPK 120:60:40 kg/ha",
    "market_price": "₹2000-2500 per quintal"
  }
}
```

**Error Response (400):**
```json
{
  "error": "Nitrogen is required"
}
```

**Error Response (500):**
```json
{
  "error": "Internal server error message"
}
```

### Parameter Ranges

| Parameter    | Min Value | Max Value | Unit  |
|--------------|-----------|-----------|-------|
| Nitrogen     | 0         | 140       | kg/ha |
| Phosphorus   | 5         | 145       | kg/ha |
| Potassium    | 5         | 205       | kg/ha |
| Temperature  | 8.825     | 43.675    | °C    |
| Humidity     | 14.258    | 99.981    | %     |
| Ph           | 3.505     | 9.935     | -     |
| Rainfall     | 20.211    | 298.56    | mm    |

### Example Usage

#### Python
```python
import requests

url = "http://localhost:5000/api/predict"
data = {
    "Nitrogen": 90,
    "Phosphorus": 42,
    "Potassium": 43,
    "Temperature": 20.87,
    "Humidity": 82.00,
    "Ph": 6.50,
    "Rainfall": 202.93
}

response = requests.post(url, json=data)
result = response.json()
print(f"Recommended Crop: {result['crop']}")
print(f"Confidence: {result['confidence']}%")
```

#### JavaScript
```javascript
fetch('http://localhost:5000/api/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    Nitrogen: 90,
    Phosphorus: 42,
    Potassium: 43,
    Temperature: 20.87,
    Humidity: 82.00,
    Ph: 6.50,
    Rainfall: 202.93
  })
})
.then(response => response.json())
.then(data => {
  console.log('Recommended Crop:', data.crop);
  console.log('Confidence:', data.confidence + '%');
});
```

#### cURL
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Nitrogen": 90,
    "Phosphorus": 42,
    "Potassium": 43,
    "Temperature": 20.87,
    "Humidity": 82.00,
    "Ph": 6.50,
    "Rainfall": 202.93
  }'
```

### Supported Crops

The system can recommend the following 22 crops:

1. Rice
2. Maize
3. Jute
4. Cotton
5. Coconut
6. Papaya
7. Orange
8. Apple
9. Muskmelon
10. Watermelon
11. Grapes
12. Mango
13. Banana
14. Pomegranate
15. Lentil
16. Blackgram
17. Mungbean
18. Mothbeans
19. Pigeonpeas
20. Kidneybeans
21. Chickpea
22. Coffee

### Rate Limiting
Currently, there are no rate limits implemented.

### Authentication
No authentication required for this version.

### Notes
- All numeric values should be within the specified ranges
- The confidence score represents the model's certainty (0-100%)
- The API returns detailed crop information including season, water needs, soil type, etc.
