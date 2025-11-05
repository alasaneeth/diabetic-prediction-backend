from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None

def load_model():
    global model, scaler
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../diabetes_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), '../scaler.pkl')
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None
        scaler = None

# Load model when module loads
load_model()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Diabetes Prediction API',
        'status': 'active',
        'endpoints': ['/health', '/predict', '/model_info']
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
        
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
        
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
            
        input_data = data['data']
        
        if len(input_data) != 8:
            return jsonify({'error': 'Expected 8 features'}), 400
            
        # Convert and predict
        input_array = np.array(input_data).reshape(1, -1)
        standardized_data = scaler.transform(input_array)
        prediction = model.predict(standardized_data)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'result': 'diabetic' if prediction == 1 else 'not diabetic',
            'message': 'The patient is likely diabetic' if prediction == 1 
                      else 'The patient is not likely diabetic'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel serverless handler
def handler(request, context):
    return app(request, context)