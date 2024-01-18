#date: 2024-01-18T16:57:44Z
#url: https://api.github.com/gists/78a3d70b694cf4e7b8d8c2f960d8b295
#owner: https://api.github.com/users/Wahid-najim

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import base64

app = Flask(__name__)

# Load barcodes from a text file
def load_valid_barcodes(file_path):
    try:
        with open(file_path, 'r') as file:
            return set(line.strip() for line in file)
    except Exception as e:
        print(f"Error loading barcodes from {file_path}: {e}")
        return set()

# Change the file name here if necessary
valid_barcodes = load_valid_barcodes('barcode.txt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    try:
        # Convert base64 image to numpy array
        encoded_data = request.data.decode('utf-8')
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'success': False, 'error': 'Image could not be decoded'})

        # Decode barcode
        barcodes = decode(img)
        if barcodes:
            barcode_data = barcodes[0].data.decode('utf-8')
            print(f"Decoded barcode: {barcode_data}")
            if barcode_data in valid_barcodes:
                return jsonify({'success': True, 'barcode': barcode_data, 'status': 'Real Product'})
            else:
                return jsonify({'success': True, 'barcode': barcode_data, 'status': 'Fake Product'})
        else:
            return jsonify({'success': False, 'error': 'No barcode found'})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
