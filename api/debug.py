from http.server import BaseHTTPRequestHandler
import json
import os
import sys

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            'message': 'Debug endpoint working',
            'python_version': sys.version,
            'current_directory': os.getcwd(),
            'files': os.listdir('.'),
            'path': self.path
        }
        
        self.wfile.write(json.dumps(response).encode())
        return