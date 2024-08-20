#date: 2024-08-20T17:09:41Z
#url: https://api.github.com/gists/e86d825d0afb42eaae80163ee40a858d
#owner: https://api.github.com/users/beccajek

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time
import logging
from flask import Flask, render_template, Response

# Initialize Flask app
app = Flask(__name__)

logging.basicConfig(filename='/home/beccajekogian/person_detection_good.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="detect.tflite", num_threads=4)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize camera
cap = cv2.VideoCapture(0)  # 0 is usually the default for the first USB camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Allow time for camera to initialize
time.sleep(2)

# GPIO setup
GPIO.setmode(GPIO.BCM)
door1Out = 14
GPIO.setup(door1Out, GPIO.OUT, initial=GPIO.HIGH)
door2Out = 24
GPIO.setup(door2Out, GPIO.OUT, initial=GPIO.HIGH)
detectionOut = 18
GPIO.setup(detectionOut, GPIO.OUT, initial=GPIO.HIGH)

doorOne_check = False
doorTwo_check = False

door1In = 15
door2In = 23

def Door1Trigger(channel):
    global doorOne_check
    global door_1_start_time
    GPIO.output(door1Out, GPIO.HIGH)
    door_1_start_time = time.time()
    print("Door 1 opened! Checking for people...")
    logging.info("Door 1 opened! Checking for people...")
    doorOne_check = True

GPIO.setup(door1In, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(door1In, GPIO.FALLING, callback=Door1Trigger, bouncetime=200)

def Door2Trigger(channel):
    global doorTwo_check
    global door_2_start_time
    GPIO.output(door2Out, GPIO.HIGH)
    door_2_start_time = time.time()
    print("Door 2 opened! Checking for people...")
    logging.info("Door 2 opened! Checking for people...")
    doorTwo_check = True

GPIO.setup(door2In, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.add_event_detect(door2In, GPIO.FALLING, callback=Door2Trigger, bouncetime=200)

def detect_person(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_shape = input_details[0]['shape']
    input_data = cv2.resize(image, (input_shape[1], input_shape[2]))
    input_data = np.expand_dims(input_data, axis=0)
    if input_details[0]['dtype'] == np.uint8:
        input_data = input_data.astype(np.uint8)
    else:
        input_data = (np.float32(input_data) - 127.5) / 127.5
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    person_detections = [(box, score) for box, class_id, score in zip(boxes, classes, scores) 
                         if class_id == 0 and score > 0.3]
    return person_detections

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.info("Failed to grab frame")
            continue
        
        detections = detect_person(frame)
                   
        if detections:
            GPIO.output(detectionOut, GPIO.LOW)
            logging.info("Person detected!")
            for box, score in detections:
                ymin, xmin, ymax, xmax = box
                (left, right, top, bottom) = (xmin * 640, xmax * 640, ymin * 480, ymax * 480)
                frame = cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
             
        else: 
            GPIO.output(detectionOut, GPIO.HIGH)
            logging.info("No one detected.")
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if doorOne_check:
            if (time.time() - door_1_start_time) > 0.25:
                if detections:
                    GPIO.output(door1Out, GPIO.LOW)
                    logging.info("Person detected in one! Door1out LOW.")
                else:
                    logging.info("No one in one! Door1out HIGH.")
                    GPIO.output(door1Out, GPIO.HIGH)
                doorOne_check = False
                
        if doorTwo_check:
            if (time.time() - door_2_start_time) > 0.25:
                if detections:
                    GPIO.output(door2Out, GPIO.LOW)
                    logging.info("Person detected in two! Door2out LOW.")
                else:
                    logging.info("No one in two! Door2out HIGH.")
                    GPIO.output(door2Out, GPIO.HIGH)
                doorTwo_check = False

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logging.info("Program stopped by user")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        time.sleep(10)
    finally:
        cap.release()
        GPIO.cleanup()