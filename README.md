# 🌾 Crop Recommendation System Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Description

The **Crop Recommendation System** is an intelligent machine learning application that provides personalized crop recommendations based on soil and climate conditions. Using advanced Random Forest algorithm with 95%+ accuracy, it helps farmers and agricultural professionals make data-driven decisions to optimize yields and maximize profitability.

### 🎯 Key Features

✨ **Smart Recommendations**
- AI-powered crop predictions with confidence scores
- Top 3 crop alternatives for better decision making
- 22 different crop types supported

🌐 **Modern Web Interface**
- Responsive design for all devices
- Dark mode support
- Interactive tooltips and animations
- Real-time form validation

📊 **Advanced Analytics**
- Prediction history tracking
- Export data to CSV
- Confidence percentage display
- Detailed crop information cards

🔌 **Developer Friendly**
- RESTful API endpoint
- JSON response format
- Comprehensive API documentation
- Easy integration with other systems

📈 **Comprehensive Crop Data**
- Growing season information
- Water requirements
- Soil type recommendations
- Temperature ranges
- Fertilizer recommendations
- Market price estimates

## 🚀 Technologies Used

- **Python 3.11+** - Core programming language
- **Flask 3.1.2** - Web framework
- **scikit-learn 1.7.1** - Machine learning library
- **pandas 2.3.2** - Data manipulation
- **NumPy 2.3.2** - Numerical computing
- **Bootstrap 5** - Frontend framework
- **Font Awesome 6** - Icons
- **Random Forest Classifier** - ML algorithm (95%+ accuracy)

## 📦 Installation and Usage

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/<your-username>/Crop-Recommendation-System-Using-Machine-Learning.git
cd Crop-Recommendation-System-Using-Machine-Learning
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv crop
```

**Activate Virtual Environment:**
- Windows: `crop\Scripts\activate`
- Linux/Mac: `source crop/bin/activate`

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python app.py
```

### Step 5: Access the Application
Open your browser and navigate to:
```
http://localhost:5000
```

## 💻 Usage Guide

### Web Interface

1. **Enter Soil Parameters:**
   - Nitrogen (N): 0-140 kg/ha
   - Phosphorus (P): 5-145 kg/ha
   - Potassium (K): 5-205 kg/ha

2. **Enter Climate Data:**
   - Temperature: 8.825-43.675°C
   - Humidity: 14.258-99.981%
   - pH: 3.505-9.935
   - Rainfall: 20.211-298.56 mm

3. **Get Recommendations:**
   - Click "Get Recommendation"
   - View best crop with confidence score
   - See top 3 alternatives
   - Read detailed crop information

### API Usage

**Endpoint:** `POST /api/predict`

**Example Request:**
```python
import requests

response = requests.post('http://localhost:5000/api/predict', json={
    "Nitrogen": 90,
    "Phosphorus": 42,
    "Potassium": 43,
    "Temperature": 20.87,
    "Humidity": 82.00,
    "Ph": 6.50,
    "Rainfall": 202.93
})

result = response.json()
print(f"Crop: {result['crop']}, Confidence: {result['confidence']}%")
```

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for complete API reference.

## 🌱 Supported Crops

The system recommends from 22 crop types:

**Cereals:** Rice, Maize  
**Fibers:** Jute, Cotton  
**Fruits:** Coconut, Papaya, Orange, Apple, Muskmelon, Watermelon, Grapes, Mango, Banana, Pomegranate  
**Pulses:** Lentil, Blackgram, Mungbean, Mothbeans, Pigeonpeas, Kidneybeans, Chickpea  
**Beverages:** Coffee

## 📊 Model Performance

- **Algorithm:** Random Forest Classifier (Hyperparameter Tuned)
- **Training Accuracy:** 99.54%
- **Test Accuracy:** 99%+ (with tuning)
- **Cross-Validation Score:** 98.92% (±0.45%)
- **F1 Score:** 0.99
- **Dataset Size:** 2,200+ samples
- **Features:** 7 base + 15 engineered = 22 total features

### 🔬 ML Enhancements
- ✅ Hyperparameter optimization (RandomizedSearchCV)
- ✅ 5-Fold cross-validation
- ✅ Feature engineering (NPK ratios, climate interactions)
- ✅ Ensemble methods (Voting Classifier)
- ✅ Comprehensive evaluation metrics
- ✅ Model explainability (Feature importance)

See [ML_IMPROVEMENTS.md](ML_IMPROVEMENTS.md) for detailed ML documentation.

## 🎨 Screenshots

### Main Interface
![Main Interface](static/screenshots/main.png)

### Prediction Results
![Results](static/screenshots/results.png)

### History Tracking
![History](static/screenshots/history.png)

## 📁 Project Structure

```
Crop-Recommendation-System/
│
├── app.py                      # Main Flask application
├── crop_info.py                # Crop information database
├── model.pkl                   # Trained ML model
├── standscaler.pkl             # Standard scaler
├── minmaxscaler.pkl           # MinMax scaler
├── Crop_recommendation.csv     # Training dataset
├── requirements.txt            # Python dependencies
├── API_DOCUMENTATION.md        # API reference
│
├── templates/
│   ├── index.html             # Main web interface
│   └── history.html           # Prediction history page
│
├── static/
│   ├── style.css              # Custom styles
│   └── images/                # Image assets
│
└── crop/                      # Virtual environment (not in repo)
```


