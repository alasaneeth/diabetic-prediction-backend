from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return jsonify({"message": "Diabetes Prediction API"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Extract features
        features = [
            data['pregnancies'],
            data['glucose'],
            data['bloodPressure'],
            data['skinThickness'],
            data['insulin'],
            data['bmi'],
            data['diabetesPedigree'],
            data['age']
        ]
        
        # Convert to numpy array and reshape
        features_array = np.asarray(features).reshape(1, -1)
        
        # Standardize the features
        std_features = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(std_features)
        
        # Prepare response
        result = "diabetic" if prediction[0] == 1 else "not diabetic"
        confidence = "Please consult with a healthcare professional for accurate diagnosis."
        
        return jsonify({
            'prediction': result,
            'message': f'The patient is {result}',
            'confidence_note': confidence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)