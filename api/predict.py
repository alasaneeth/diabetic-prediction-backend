from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('model/diabetes_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features from request
        features = [
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['bloodPressure']),
            float(data['skinThickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['diabetesPedigree']),
            float(data['age'])
        ]
        
        # Convert to numpy array and reshape
        input_data = np.asarray(features).reshape(1, -1)
        
        # Standardize the data
        std_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(std_data)
        
        result = {
            'prediction': int(prediction[0]),
            'message': 'The patient is diabetic' if prediction[0] == 1 else 'The patient is not diabetic'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)