#date: 2025-12-16T17:02:31Z
#url: https://api.github.com/gists/8aa3ff84cf1c7c444c5327131e854510
#owner: https://api.github.com/users/markk0042

cd /opt/Park-Wise/anpr-service
python3 << 'ENDPYTHON'
content = '''"""
Real-time ANPR (Automatic Number Plate Recognition) Service
Uses YOLO for license plate detection and EasyOCR for text recognition
"""

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
import io
import os
import urllib.request

app = Flask(__name__)
CORS(app)

LICENSE_PLATE_MODEL_URLS = [
    'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt',
]

def download_model(url, dest_path):
    """Download a model file from URL"""
    try:
        print(f"Downloading license plate model from {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Model downloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False

print("Loading YOLO model...")
try:
    import torch
    if hasattr(torch.serialization, 'add_safe_globals'):
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
        torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
    
    model_path = os.getenv('YOLO_MODEL_PATH', None)
    
    if not model_path:
        if os.path.exists('lp_yolo.pt'):
            model_path = 'lp_yolo.pt'
        elif os.path.exists('license_plate_yolo.pt'):
            model_path = 'license_plate_yolo.pt'
        elif os.path.exists('license_plate_detector.pt'):
            model_path = 'license_plate_detector.pt'
        elif os.path.exists('yolov8n.pt'):
            model_path = 'yolov8n.pt'
        else:
            model_path = 'yolov8n.pt'
            if not os.path.exists(model_path):
                print("Downloading YOLOv8n model (general purpose)...")
                download_model('https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt', model_path)
    
    model = YOLO(model_path, task='detect')
    print(f"YOLO model loaded: {model_path}")
    if 'lp_yolo' in model_path or 'license_plate' in model_path:
        print("✓ Using license plate-specific model - optimized for plate detection!")
    else:
        print("Note: For best results, use a license plate-specific model trained on your region's plates.")
        print("You can download one from Roboflow, HuggingFace, or train your own.")
except Exception as e:
    print(f"Warning: Could not load YOLO model: {e}")
    print("Using basic image processing instead")
    model = None

print("Initializing EasyOCR...")
try:
    reader = easyocr.Reader(['en'], gpu=False, quantize=True, model_storage_directory='/tmp/easyocr')
    print("EasyOCR ready!")
except Exception as e:
    print(f"Warning: EasyOCR initialization failed: {e}")
    print("Trying without quantization...")
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR ready!")
    except Exception as e2:
        print(f"Error: Could not initialize EasyOCR: {e2}")
        reader = None

def preprocess_image(image):
    """Preprocess image for better OCR results"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return cleaned

def detect_license_plate_yolo(image):
    """Detect license plates using YOLO"""
    if model is None:
        return []
    
    results = model(image, conf=0.25, iou=0.45, imgsz=640)
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            area = width * height
            
            is_valid = False
            
            if confidence > 0.25:
                if aspect_ratio >= 1.5 and aspect_ratio <= 6.0 and area > 400:
                    is_valid = True
            
            if is_valid:
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': confidence
                })
    
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections

def detect_license_plate_contours(image):
    """Fallback: Detect license plates using contour detection"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        area = cv2.contourArea(contour)
        
        if 1.5 <= aspect_ratio <= 7.0 and area > 500:
            detections.append({
                'bbox': [x, y, x + w, y + h],
                'confidence': 0.5
            })
    
    return detections

def extract_text_from_roi(image, bbox):
    """Extract text from a region of interest"""
    x1, y1, x2, y2 = bbox
    
    padding = 10
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    
    roi = image[y1:y2, x1:x2]
    
    if roi.size == 0:
        return None, 0.0
    
    if reader is None:
        return None, 0.0

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    h, w = roi_rgb.shape[:2]
    if h < 40 or w < 100:
        roi_rgb = cv2.resize(
            roi_rgb,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        )

    results = reader.readtext(roi_rgb)
    if not results:
        gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            15,
            10,
        )
        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
        results = reader.readtext(thresh_rgb)

    try:
        raw_texts = [r[1] for r in results]
        print(f"[ANPR OCR] ROI {bbox} -> {raw_texts}")
    except Exception:
        pass
    
    if not results:
        return None, 0.0
    
    text_parts = []
    total_confidence = 0.0
    
    for (bbox, text, confidence) in results:
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        if cleaned:
            text_parts.append(cleaned)
            total_confidence += confidence
    
    if not text_parts:
        return None, 0.0
    
    combined_text = ''.join(text_parts)
    avg_confidence = total_confidence / len(results)
    
    irish_pattern = r'^[0-9]{1,2}[A-Z][0-9]{4,6}$'
    uk_pattern = r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}$'
    european_pattern = r'^[A-Z0-9]{2,10}$'
    flexible_pattern = r'^[A-Z0-9]{2,10}$'
    
    if (re.match(irish_pattern, combined_text) or 
        re.match(uk_pattern, combined_text) or 
        re.match(european_pattern, combined_text) or
        re.match(flexible_pattern, combined_text)):
        return combined_text, avg_confidence
    
    if len(combined_text) >= 3 and len(combined_text) <= 12:
        return combined_text, avg_confidence * 0.8
    
    return combined_text, avg_confidence

def process_image(image_array):
    """Main processing function"""
    if len(image_array.shape) == 2:
        image = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    else:
        image = image_array
    
    detections = []
    
    if model is not None:
        detections = detect_license_plate_yolo(image)
    
    if not detections:
        detections = detect_license_plate_contours(image)
    
    results = []
    
    for detection in detections:
        bbox = detection['bbox']
        text, confidence = extract_text_from_roi(image, bbox)
        
        if text and len(text) >= 2:
            results.append({
                'registration': text,
                'confidence': float(confidence),
                'bbox': bbox
            })
    
    if not results:
        text, confidence = extract_text_from_roi(image, [0, 0, image.shape[1], image.shape[0]])
        if text and len(text) >= 2:
            results.append({
                'registration': text,
                'confidence': float(confidence),
                'bbox': [0, 0, image.shape[1], image.shape[0]]
            })

    print(f"[ANPR] Final detections: {results}")
    return results

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'ocr_ready': reader is not None
    })

@app.route('/process', methods=['POST'])
def process():
    """Process an image and return detected license plates"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image)
        
        results = process_image(image_array)
        
        return jsonify({
            'success': True,
            'detections': results,
            'count': len(results)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/process-batch', methods=['POST'])
def process_batch():
    """Process multiple images"""
    try:
        data = request.get_json()
        
        if 'images' not in data or not isinstance(data['images'], list):
            return jsonify({'error': 'No images array provided'}), 400
        
        all_results = []
        
        for image_data in data['images']:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            
            results = process_image(image_array)
            all_results.extend(results)
        
        return jsonify({
            'success': True,
            'detections': all_results,
            'count': len(all_results)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"Starting ANPR service on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
'''
with open('anpr.py', 'w') as f:
    f.write(content)
print("File anpr.py created successfully!")
ENDPYTHON