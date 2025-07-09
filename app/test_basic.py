#!/usr/bin/env python3
import os

# Test 1: Pure Python web server (no FastAPI)
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class TestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        response = {
            "message": "Basic Python HTTP server",
            "port": os.getenv("PORT", "8000"),
            "path": self.path
        }
        self.wfile.write(json.dumps(response).encode())

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    server = HTTPServer(('0.0.0.0', port), TestHandler)
    print(f"Starting basic HTTP server on port {port}")
    server.serve_forever()