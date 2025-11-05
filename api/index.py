from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import traceback

# Initialize variables
DEPENDENCIES_LOADED = False
MODEL_LOADED = False
LOAD_ERROR = None

try:
    import joblib
    import numpy as np
    DEPENDENCIES_LOADED = True
    print("✅ Dependencies loaded successfully")
except ImportError as e:
    DEPENDENCIES_LOADED = False
    LOAD_ERROR = f"Dependencies error: {str(e)}"
    print(f"❌ {LOAD_ERROR}")

def load_model():
    global MODEL_LOADED, LOAD_ERROR
    if not DEPENDENCIES_LOADED:
        LOAD_ERROR = "Dependencies not loaded"
        return False
        
    try:
        print("🔍 Checking directory contents...")
        current_dir = os.getcwd()
        files = os.listdir(current_dir)
        print(f"📁 Current directory: {current_dir}")
        print(f"📁 Files available: {files}")
        
        # Check for model files with different possible names
        model_files = [
            'diabetes_model.pkl', 
            'scaler.pkl',
            'diabetes_model_compatible.pkl',
            'scaler_compatible.pkl'
        ]
        
        found_files = [f for f in model_files if f in files]
        print(f"🔍 Found model files: {found_files}")
        
        if 'diabetes_model.pkl' in files and 'scaler.pkl' in files:
            print("📦 Loading model files...")
            
            # Get file sizes
            model_size = os.path.getsize('diabetes_model.pkl')
            scaler_size = os.path.getsize('scaler.pkl')
            print(f"📊 Model file size: {model_size} bytes")
            print(f"📊 Scaler file size: {scaler_size} bytes")
            
            # Check if files are too large (Vercel limit ~50MB)
            if model_size > 45 * 1024 * 1024:  # 45MB
                LOAD_ERROR = f"Model file too large: {model_size} bytes"
                return False
                
            # Load the files
            global model, scaler
            model = joblib.load('diabetes_model.pkl')
            scaler = joblib.load('scaler.pkl')
            
            print(f"✅ Model type: {type(model)}")
            print(f"✅ Scaler type: {type(scaler)}")
            MODEL_LOADED = True
            return True
        else:
            LOAD_ERROR = f"Model files not found. Available: {files}"
            return False
            
    except Exception as e:
        LOAD_ERROR = f"Model loading error: {str(e)}"
        print(f"❌ {LOAD_ERROR}")
        print(f"🔍 Traceback: {traceback.format_exc()}")
        return False

# Attempt to load model
if DEPENDENCIES_LOADED:
    load_model()

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
                'load_error': LOAD_ERROR
            }
        elif self.path == '/debug':
            response = {
                'python_version': sys.version,
                'current_directory': os.getcwd(),
                'files_in_directory': os.listdir('.'),
                'dependencies_loaded': DEPENDENCIES_LOADED,
                'model_loaded': MODEL_LOADED,
                'load_error': LOAD_ERROR,
                'environment_variables': dict(os.environ)
            }
        else:
            response = {
                'message': 'Diabetes Prediction API',
                'status': 'running',
                'model_ready': MODEL_LOADED,
                'endpoints': {
                    'GET /health': 'Check API health',
                    'GET /debug': 'Debug information',
                    'POST /predict': 'Make prediction'
                }
            }
        
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {}
        
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data)
            else:
                data = {}
            
            if self.path == '/predict':
                if not MODEL_LOADED:
                    response = {
                        'error': 'Model not available',
                        'details': LOAD_ERROR,
                        'dependencies_loaded': DEPENDENCIES_LOADED,
                        'model_loaded': MODEL_LOADED
                    }
                else:
                    input_data = data.get('data', [])
                    
                    if not input_data or len(input_data) != 8:
                        response = {
                            'error': 'Invalid input',
                            'expected': '8 numeric features',
                            'received': f'{len(input_data)} features' if input_data else 'no data'
                        }
                    else:
                        # Make prediction
                        input_array = np.array(input_data, dtype=float).reshape(1, -1)
                        standardized_data = scaler.transform(input_array)
                        prediction = model.predict(standardized_data)
                        
                        response = {
                            'prediction': int(prediction[0]),
                            'result': 'diabetic' if prediction[0] == 1 else 'not diabetic',
                            'confidence': 'high',  # You can add probability if available
                            'status': 'success'
                        }
            else:
                response = {'error': 'Endpoint not found'}
                
        except Exception as e:
            response = {
                'error': 'Processing failed',
                'details': str(e),
                'model_loaded': MODEL_LOADED
            }
        
        self.wfile.write(json.dumps(response, indent=2).encode())
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()