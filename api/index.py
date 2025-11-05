from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import traceback

try:
    import joblib
    import numpy as np
    DEPENDENCIES_LOADED = True
except ImportError as e:
    DEPENDENCIES_LOADED = False
    IMPORT_ERROR = str(e)

# Global variables
model = None
scaler = None

def load_model():
    global model, scaler
    try:
        print("🔍 Checking for model files...")
        files = os.listdir('.')
        print(f"📁 Available files: {files}")
        
        if 'diabetes_model.pkl' in files and 'scaler.pkl' in files:
            print("✅ Model files found, loading...")
            model = joblib.load('diabetes_model.pkl')
            scaler = joblib.load('scaler.pkl')
            print("✅ Model loaded successfully!")
            return True
        else:
            print("❌ Model files not found")
            return False
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# Try to load model
MODEL_LOADED = load_model() if DEPENDENCIES_LOADED else False

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        if self.path == '/health':
            response = {
                'status': 'healthy',
                'dependencies_loaded': DEPENDENCIES_LOADED,
                'model_loaded': MODEL_LOADED,
                'python_version': sys.version
            }
        elif self.path == '/debug':
            response = {
                'python_version': sys.version,
                'current_directory': os.getcwd(),
                'files_in_directory': os.listdir('.'),
                'dependencies_loaded': DEPENDENCIES_LOADED,
                'model_loaded': MODEL_LOADED,
                'import_error': IMPORT_ERROR if not DEPENDENCIES_LOADED else None
            }
        else:
            response = {
                'message': 'Diabetes Prediction API',
                'status': 'running',
                'endpoints': ['/health', '/debug', '/predict'],
                'model_ready': MODEL_LOADED
            }
        
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            
            if self.path == '/predict':
                if not DEPENDENCIES_LOADED:
                    response = {
                        'error': 'Dependencies not loaded',
                        'details': IMPORT_ERROR
                    }
                elif not MODEL_LOADED:
                    response = {
                        'error': 'Model not loaded',
                        'available_files': os.listdir('.')
                    }
                else:
                    # Simple prediction logic
                    input_data = data.get('data', [])
                    
                    if len(input_data) != 8:
                        response = {
                            'error': f'Expected 8 features, got {len(input_data)}'
                        }
                    else:
                        try:
                            input_array = np.array(input_data, dtype=float).reshape(1, -1)
                            standardized_data = scaler.transform(input_array)
                            prediction = model.predict(standardized_data)
                            
                            response = {
                                'prediction': int(prediction[0]),
                                'result': 'diabetic' if prediction[0] == 1 else 'not diabetic',
                                'status': 'success'
                            }
                        except Exception as e:
                            response = {
                                'error': 'Prediction failed',
                                'details': str(e)
                            }
            else:
                response = {'error': 'Endpoint not found'}
                
        except Exception as e:
            response = {
                'error': 'Request processing failed',
                'details': str(e)
            }
        
        self.wfile.write(json.dumps(response).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()