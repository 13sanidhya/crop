from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import logging
from datetime import datetime
import warnings
from crop_info import CROP_INFO

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Load model and scalers
try:
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Error loading models: {str(e)}")
    raise

app = Flask(__name__)

# Store prediction history
prediction_history = []

# Dataset ranges
ranges = {
    'Nitrogen': (0, 140),
    'Phosphorus': (5, 145),   # spelling must match HTML form
    'Potassium': (5, 205),
    'Temperature': (8.825, 43.675),
    'Humidity': (14.258, 99.981),
    'Ph': (3.505, 9.935),
    'Rainfall': (20.211, 298.56)
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and validate inputs
        feature_list = []
        input_data = {}
        
        for key, (min_val, max_val) in ranges.items():
            value = request.form.get(key)
            if value is None or value.strip() == "":
                logging.warning(f"Missing input: {key}")
                return render_template('index.html', 
                                     error=f"{key} is required. Please fill all fields.")
            try:
                value = float(value)
                input_data[key] = value
            except ValueError:
                logging.warning(f"Invalid input for {key}: {value}")
                return render_template('index.html', 
                                     error=f"{key} must be a valid number.")
            if not (min_val <= value <= max_val):
                logging.warning(f"{key} out of range: {value}")
                return render_template('index.html', 
                                     error=f"{key} must be between {min_val} and {max_val}.")
            feature_list.append(value)

        # Convert to array for model
        single_pred = np.array(feature_list).reshape(1, -1)

        # Scaling
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Get prediction with probability
        prediction = model.predict(final_features)
        probabilities = model.predict_proba(final_features)[0]
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probabilities)[-3:][::-1]
        top_3_crops = []
        
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
            7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
            12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
            21: "Chickpea", 22: "Coffee"
        }
        
        for idx in top_3_idx:
            crop_num = idx + 1
            if crop_num in crop_dict:
                confidence = probabilities[idx] * 100
                crop_name = crop_dict[crop_num]
                crop_details = CROP_INFO.get(crop_num, {})
                top_3_crops.append({
                    'name': crop_name,
                    'confidence': round(confidence, 2),
                    'info': crop_details
                })

        best_crop = crop_dict.get(prediction[0])
        confidence = probabilities[prediction[0] - 1] * 100
        
        # Store in history
        prediction_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'inputs': input_data,
            'prediction': best_crop,
            'confidence': round(confidence, 2)
        }
        prediction_history.append(prediction_entry)
        
        logging.info(f"Prediction: {best_crop} with {confidence:.2f}% confidence")
        
        if best_crop:
            result = f"{best_crop}"
            crop_info = CROP_INFO.get(prediction[0], {})
            return render_template('index.html', 
                                 result=result,
                                 confidence=round(confidence, 2),
                                 crop_info=crop_info,
                                 top_3_crops=top_3_crops,
                                 inputs=input_data)
        else:
            return render_template('index.html', 
                                 error="Could not determine the best crop. Please check your inputs.")

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return render_template('index.html', 
                             error=f"An error occurred: {str(e)}")

# API endpoint for JSON response
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json()
        
        feature_list = []
        for key, (min_val, max_val) in ranges.items():
            value = data.get(key)
            if value is None:
                return jsonify({'error': f'{key} is required'}), 400
            if not (min_val <= value <= max_val):
                return jsonify({'error': f'{key} out of range'}), 400
            feature_list.append(float(value))
        
        single_pred = np.array(feature_list).reshape(1, -1)
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)
        
        prediction = model.predict(final_features)
        probabilities = model.predict_proba(final_features)[0]
        
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
            7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
            12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
            21: "Chickpea", 22: "Coffee"
        }
        
        crop = crop_dict.get(prediction[0])
        confidence = probabilities[prediction[0] - 1] * 100
        
        return jsonify({
            'success': True,
            'crop': crop,
            'confidence': round(confidence, 2),
            'crop_info': CROP_INFO.get(prediction[0], {})
        })
    
    except Exception as e:
        logging.error(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# History endpoint
@app.route('/history')
def history():
    return render_template('history.html', history=prediction_history)

# Clear history
@app.route('/clear_history')
def clear_history():
    global prediction_history
    prediction_history = []
    logging.info("Prediction history cleared")
    return render_template('history.html', history=prediction_history, message="History cleared successfully!")


if __name__ == "__main__":
    app.run(debug=True)
