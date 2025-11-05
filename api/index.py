from flask import Flask, request, jsonify
import json
import sys
import os
import traceback

print("🚀 Starting Flask application on Vercel...")

app = Flask(__name__)

# Try to import dependencies with error handling
try:
    import joblib
    import numpy as np
    from flask_cors import CORS
    CORS(app)
    print("✅ All dependencies imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Installed packages:", os.listdir('.'))
    # Continue without CORS for now

# Global variables
model = None
scaler = None

def load_model():
    global model, scaler
    try:
        print("📦 Attempting to load model files...")
        
        # List files in current directory
        current_files = os.listdir('.')
        print("📁 Files in directory:", current_files)
        
        # Check if model files exist
        if 'diabetes_model.pkl' not in current_files:
            print("❌ diabetes_model.pkl not found!")
            return
        if 'scaler.pkl' not in current_files:
            print("❌ scaler.pkl not found!")
            return
            
        print("✅ Model files found, loading...")
        
        # Load model and scaler
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        print("✅ Model and scaler loaded successfully!")
        print(f"📊 Model type: {type(model)}")
        print(f"📊 Scaler type: {type(scaler)}")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        model = None
        scaler = None

# Load model on startup
load_model()

@app.route('/')
def home():
    return jsonify({
        'message': 'Diabetes Prediction API is running!',
        'status': 'active',
        'model_loaded': model is not None,
        'endpoints': ['/health', '/predict', '/debug']
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/debug')
def debug():
    """Debug endpoint to check server status"""
    return jsonify({
        'python_version': sys.version,
        'current_directory': os.getcwd(),
        'files_in_directory': os.listdir('.'),
        'model_loaded': model is not None,
        'model_type': str(type(model)) if model else None,
        'scaler_loaded': scaler is not None
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Make diabetes prediction"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        print("📥 Received prediction request")
        
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded', 
                'model_status': model is None,
                'scaler_status': scaler is None
            }), 503
        
        data = request.get_json()
        print(f"📊 Request data: {data}")
        
        if not data or 'data' not in data:
            return jsonify({'error': 'Missing data field'}), 400
            
        input_data = data['data']
        
        if not isinstance(input_data, list):
            return jsonify({'error': 'Data must be a list'}), 400
            
        if len(input_data) != 8:
            return jsonify({
                'error': f'Expected 8 features, got {len(input_data)}'
            }), 400
        
        # Convert to numpy array
        input_array = np.array(input_data, dtype=float).reshape(1, -1)
        print(f"🔢 Input array: {input_array}")
        
        # Standardize and predict
        standardized_data = scaler.transform(input_array)
        prediction = model.predict(standardized_data)
        
        result = int(prediction[0])
        print(f"🎯 Prediction result: {result}")
        
        return jsonify({
            'prediction': result,
            'result': 'diabetic' if result == 1 else 'not diabetic',
            'message': 'The patient is likely diabetic' if result == 1 
                      else 'The patient is not likely diabetic',
            'status': 'success'
        })
        
    except Exception as e:
        print(f"💥 Prediction error: {str(e)}")
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

# Vercel serverless function handler
def handler(request, context):
    print("🔄 Vercel handler called")
    return app(request, context)

print("✅ Flask app setup complete")