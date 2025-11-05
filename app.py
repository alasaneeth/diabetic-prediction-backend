from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # This allows React app to communicate with Flask

# Global variables to store the model and scaler
model = None
scaler = None

def load_model():
    """
    Load the machine learning model and scaler
    """
    global model, scaler
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("✅ Model and scaler loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        scaler = None

# Load model when the application starts
load_model()

@app.route('/')
def home():
    return jsonify({
        'message': 'Diabetes Prediction API is running!',
        'endpoints': {
            'GET /health': 'Check API health and model status',
            'POST /predict': 'Make diabetes prediction',
            'GET /model_info': 'Get model information'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify API and model status
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': str(type(model)),
        'features_expected': 8,
        'feature_names': [
            'Pregnancies',
            'Glucose',
            'BloodPressure',
            'SkinThickness',
            'Insulin',
            'BMI',
            'DiabetesPedigreeFunction',
            'Age'
        ]
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make diabetes prediction based on input data
    Expected input format:
    {
        "data": [1, 89, 66, 23, 94, 28.1, 0.167, 21]
    }
    """
    # Check if model is loaded
    if model is None or scaler is None:
        return jsonify({'error': 'Prediction model not available'}), 503
    
    try:
        # Get JSON data from request
        request_data = request.get_json()
        
        # Validate input
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'Missing "data" field in request'}), 400
        
        input_data = request_data['data']
        
        # Validate input data type and length
        if not isinstance(input_data, list):
            return jsonify({'error': 'Input data must be a list'}), 400
        
        if len(input_data) != 8:
            return jsonify({
                'error': f'Expected 8 features, got {len(input_data)}',
                'expected_features': 8
            }), 400
        
        # Convert to numpy array and validate numeric values
        try:
            input_array = np.array(input_data, dtype=float).reshape(1, -1)
        except ValueError:
            return jsonify({'error': 'All input values must be numeric'}), 400
        
        # Standardize the input data
        standardized_data = scaler.transform(input_array)
        
        # Make prediction
        prediction = model.predict(standardized_data)
        prediction_proba = model.decision_function(standardized_data)
        
        # Prepare response
        result = int(prediction[0])
        confidence = float(prediction_proba[0])
        
        response = {
            'prediction': result,
            'confidence_score': abs(confidence),
            'result': 'diabetic' if result == 1 else 'not diabetic',
            'message': 'The patient is likely diabetic' if result == 1 
                      else 'The patient is not likely diabetic'
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Make predictions for multiple patients at once
    Expected input format:
    {
        "data": [
            [1, 89, 66, 23, 94, 28.1, 0.167, 21],
            [2, 85, 66, 29, 0, 26.6, 0.351, 31]
        ]
    }
    """
    if model is None or scaler is None:
        return jsonify({'error': 'Prediction model not available'}), 503
    
    try:
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'Missing "data" field in request'}), 400
        
        batch_data = request_data['data']
        
        if not isinstance(batch_data, list) or not all(isinstance(item, list) for item in batch_data):
            return jsonify({'error': 'Input data must be a list of lists'}), 400
        
        # Validate each input
        for i, item in enumerate(batch_data):
            if len(item) != 8:
                return jsonify({
                    'error': f'Item {i}: Expected 8 features, got {len(item)}'
                }), 400
        
        # Convert to numpy array
        try:
            input_array = np.array(batch_data, dtype=float)
        except ValueError:
            return jsonify({'error': 'All input values must be numeric'}), 400
        
        # Standardize and predict
        standardized_data = scaler.transform(input_array)
        predictions = model.predict(standardized_data)
        confidence_scores = model.decision_function(standardized_data)
        
        # Prepare batch response
        results = []
        for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
            results.append({
                'patient_id': i + 1,
                'prediction': int(pred),
                'confidence_score': abs(float(conf)),
                'result': 'diabetic' if pred == 1 else 'not diabetic'
            })
        
        return jsonify({
            'batch_results': results,
            'total_patients': len(results),
            'diabetic_count': sum(pred == 1 for pred in predictions),
            'non_diabetic_count': sum(pred == 0 for pred in predictions)
        })
        
    except Exception as e:
        print(f"Batch prediction error: {e}")
        return jsonify({'error': 'Internal server error during batch prediction'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Diabetes Prediction API...")
    print("📊 Model status:", "Loaded" if model is not None else "Not loaded")
    print("🔧 Starting server on http://localhost:5000")
    
    # Run the application
    app.run(
        host='0.0.0.0',  # Allows external connections
        port=5000,
        debug=True       # Set to False in production
    )