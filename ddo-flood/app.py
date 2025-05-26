from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from attack_detector import detect_attack
import logging
from collections import deque
from time import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Store blocked IPs
blocked_ips = set()

# Store request history for each IP
request_history = {}

@app.route('/')
def index():
    client_ip = request.remote_addr
    if client_ip in blocked_ips:
        return render_template('blocked.html')
    return render_template('index.html')

@app.route('/blocked')
def blocked():
    client_ip = request.remote_addr
    if client_ip not in blocked_ips:
        return render_template('index.html')
    return render_template('blocked.html')

@app.route('/block_ip', methods=['POST'])
def block_ip():
    client_ip = request.remote_addr
    blocked_ips.add(client_ip)
    logger.warning(f"Blocked IP address: {client_ip}")
    return jsonify({"status": "ok", "message": f"IP {client_ip} has been blocked", "ip": client_ip}), 200

@app.route('/send_requests.js')
def send_requests_js():
    return app.send_static_file('send_requests.js')

@app.route('/receive', methods=['POST'])
def receive_data():
    client_ip = request.remote_addr
    if client_ip in blocked_ips:
        return jsonify({"status": "error", "message": "Your IP has been blocked due to suspicious activity"}), 403

    try:
        # Get the data from the request
        data = request.get_json()
        if not data or not isinstance(data, list):
            return jsonify({"status": "error", "message": "Invalid data format"}), 400

        # Initialize request history for this IP if not exists
        if client_ip not in request_history:
            request_history[client_ip] = {
                'count': 0,
                'start_time': time(),
                'requests': []
            }

        # Update request history
        request_history[client_ip]['count'] += 1
        request_history[client_ip]['requests'].extend(data)

        # Only check for attacks after minimum number of requests (600-700)
        min_requests = 600
        max_requests = 800
        
        if request_history[client_ip]['count'] >= min_requests:
            # Use the trained model to detect attacks on the last batch of requests
            batch_size = 100  # Analyze last 100 requests
            start_idx = max(0, len(request_history[client_ip]['requests']) - batch_size)
            recent_requests = request_history[client_ip]['requests'][start_idx:]
            
            result = detect_attack(recent_requests)
            
            # If attack is detected, block the IP
            if result.get("attack_detected"):
                blocked_ips.add(client_ip)
                logger.warning(f"Attack detected from IP {client_ip} after {request_history[client_ip]['count']} requests! Blocking IP.")
                result["message"] = "Attack detected and IP blocked"
                return jsonify(result), 200

            # If we've reached max requests without detection, clear history
            if request_history[client_ip]['count'] >= max_requests:
                request_history[client_ip] = {
                    'count': 0,
                    'start_time': time(),
                    'requests': []
                }

        return jsonify({"status": "ok", "message": "No attack detected"}), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"status": "error", "message": f"Error processing request: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
