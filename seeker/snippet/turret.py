#date: 2023-09-15T16:54:59Z
#url: https://api.github.com/gists/efe892120065fe94dade1eacee20fae8
#owner: https://api.github.com/users/matt-desmarais

#!/usr/bin/python3
import RPi.GPIO as GPIO
import os
import signal
import atexit
import random
from PIL import Image
import time
from picamera2 import MappedArray, Picamera2, Preview
import cv2
import libcamera
import numpy as np
import tflite_runtime.interpreter as tflite
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
import datetime
import math
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput
from servo import Servo
width, height = 1920, 1080
global firing
firing = 0
global pan
global tilt
pan = 0
tilt = 0
from pid import PIDController

buzzer_pin = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(buzzer_pin, GPIO.OUT)
GPIO.output(buzzer_pin, GPIO.LOW)

GPIO.setmode(GPIO.BCM)

panServo = Servo(pin=13, min_angle=-90, max_angle=90) # pan_servo_pin (BCM)
tiltServo = Servo(pin=12, min_angle=-90, max_angle=30) # be careful to limit the angle of the steering gear
panServo.set_angle(pan)
tiltServo.set_angle(tilt)
time.sleep(1)
global kp
kp = 0.02  # Reduced Proportional gain further

global targetIndex
targetIndex = 0
global targetHitCounter
targetHitCounter = 0
global counter
counter = 0
#for FPS calculations
global t1
t1=1
global t3
t3=1
# Initialize TensorFlow Lite interpreter
interpreter = Interpreter(model_path='/home/pi/Desktop/Sample_TFLite_model/models/edgetpu.tflite', experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
# Load the label map
with open('/home/pi/Desktop/Sample_TFLite_model/SSDMobileNetv2/coco_labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
output_tensor = interpreter.get_tensor(output_details[0]['index'])

# Initialize frame rate calculation
global frame_rate_calc
global freq
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize frame rate calculation
global frame_rate_calc2
global freq2
frame_rate_calc2 = 1
freq2 = cv2.getTickFrequency()

#width, height = 1280, 720
global targets

# Define font and color
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
font_color = (255, 255, 255)  # White
font_thickness = 2

global crosshairX
global crosshairY
crosshairW = 150
crosshairX = width
crosshairY = height
global crosshairCol
crosshairCol = (0,255,0)

# Original resolution
original_width = 640  # Replace with your original width
original_height = 480  # Replace with your original height

# New resolution
new_width = width
new_height = height

def intersection_over_union(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def non_maximum_suppression(boxes, classes, scores, iou_threshold=0.5, score_threshold=0.25):
    #print("Initial number of boxes:", len(boxes))
    
    # If there's only one box, return it
    if len(boxes) == 1:
        #print("Only one box detected.")
        if scores[0] >= score_threshold:
            #print("Box score is above threshold. Returning the box.")
            return boxes, classes, scores
        else:
            #print("Box score is below threshold. Returning empty lists.")
            return [], [], []

    # Filter out boxes with low scores
    indices = [i for i, s in enumerate(scores) if s > score_threshold]
    boxes = [boxes[i] for i in indices]
    classes = [classes[i] for i in indices]
    scores = [scores[i] for i in indices]
    
    #print("Number of boxes after score threshold filtering:", len(boxes))

    # Sort the boxes by scores in descending order
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    keep = []
    while sorted_indices:
        i = sorted_indices.pop(0)
        keep.append(i)

        # Compute the IoU for the current box with the rest
        ious = [intersection_over_union(boxes[i], boxes[j]) for j in sorted_indices]

        # Filter out boxes that have high overlap with the current box
        sorted_indices = [sorted_indices[j] for j, iou in enumerate(ious) if iou <= iou_threshold]

    #print("Number of boxes after NMS:", len(keep))

    # Gather the boxes, classes, and scores using the selected indices
    selected_boxes = [boxes[i] for i in keep]
    selected_classes = [classes[i] for i in keep]
    selected_scores = [scores[i] for i in keep]

    return selected_boxes, selected_classes, selected_scores


def get_leftmost_object_index(bounding_boxes):
    """
    Returns the index of the bounding box that's the left-most in the frame.
    
    Parameters:
    - bounding_boxes: list of bounding boxes in the format [ymin, xmin, ymax, xmax].
    
    Returns:
    - int: Index of the left-most bounding box.
    """
    
    # Extract xmin values for all bounding boxes
    xmin_values = [box[1] for box in bounding_boxes]

    # Get the index of the smallest xmin value
    leftmost_index = xmin_values.index(min(xmin_values))
    print("Left most index: "+str(leftmost_index))
    return leftmost_index

def get_rightmost_object_index(bounding_boxes):
    """
    Returns the index of the bounding box that's the right-most in the frame.
    
    Parameters:
    - bounding_boxes: list of bounding boxes in the format [ymin, xmin, ymax, xmax].
    
    Returns:
    - int: Index of the right-most bounding box.
    """
    
    # Extract xmax values for all bounding boxes
    xmax_values = [box[3] for box in bounding_boxes]

    # Get the index of the largest xmax value
    rightmost_index = xmax_values.index(max(xmax_values))
    print("Right most index: "+str(rightmost_index))
    return rightmost_index

def handle_sigstp(signum, frame):
    print("\nCtrl+Z pressed! Executing custom code...")

# Register the handler for SIGTSTP (Ctrl+Z)
signal.signal(signal.SIGTSTP, handle_sigstp)

def goodbye():
    print("Goodbye")

atexit.register(goodbye)

def get_second_closest_object_index(bounding_boxes, width, height):
    """
    Returns the index of the bounding box that's the second closest to the center of the frame.
    
    Parameters:
    - bounding_boxes: list of bounding boxes in the format [ymin, xmin, ymax, xmax].
    - width: Width of the frame.
    - height: Height of the frame.
    
    Returns:
    - int: Index of the second closest bounding box.
    """
    
    # Define a reference point (e.g., center of the frame)
    reference_point = np.array([width // 2, height // 2])

    # Calculate distances of all bounding boxes to the center
    distances = []
    for box in bounding_boxes:
        ymin, xmin, ymax, xmax = box
        center_x = (xmin + xmax) / 2 * width
        center_y = (ymin + ymax) / 2 * height
        distance = np.linalg.norm(reference_point - np.array([center_x, center_y]))
        distances.append(distance)

    # Sort distances and get the index of the second smallest distance
    sorted_indices = np.argsort(distances)
    second_closest_index = sorted_indices[1]

    return second_closest_index

def get_closest_object_index(bounding_boxes, width, height):
    """
    Returns the index of the bounding box closest to the center of the frame.
    
    Parameters:
    - bounding_boxes: list of bounding boxes in the format [ymin, xmin, ymax, xmax].
    - frame_shape: tuple representing the shape of the frame (height, width).
    
    Returns:
    - int: Index of the closest bounding box.
    """
    
    # Define a reference point (e.g., center of the frame)
    reference_point = np.array([width//2, height//2])

    # Find the index of the closest bounding box
    min_distance = float('inf')
    closest_index = -1
    for index, box in enumerate(bounding_boxes):
        ymin, xmin, ymax, xmax = box
        center_x = (xmin + xmax) / 2 * width
        center_y = (ymin + ymax) / 2 * height
        distance = np.linalg.norm(reference_point - np.array([center_x, center_y]))
        if distance < min_distance:
            min_distance = distance
            closest_index = index

    return closest_index

def get_extreme_coordinates(array):
    # Initialize with the first set of coordinates
    max_x1 = arrays[0][0]
    max_y1 = arrays[0][1]
    min_x2 = arrays[0][2]
    min_y2 = arrays[0][3]

    # Iterate through the rest of the arrays to find the extremes
    for coords in arrays[1:]:
        max_x1 = max(max_x1, coords[0])
        max_y1 = max(max_y1, coords[1])
        min_x2 = min(min_x2, coords[2])
        min_y2 = min(min_y2, coords[3])

    return max_x1, max_y1, min_x2, min_y2

# Calculate scaling factors
x_scale = new_width / original_width
y_scale = new_height / original_height
def is_crosshair_in_center(crosshair_x, crosshair_y, box_x1, box_y1, box_x2, box_y2):
    global targetHitCounter
    offsetX = (box_x2-box_x1)//2-((box_x2-box_x1)//6)
    offsetY = (box_y2-box_y1)//2-((box_y2-box_y1)//6)
    if (box_x1+offsetX) < crosshair_x//2 < (box_x2-offsetX) and (box_y1+offsetY) < crosshair_y//2 < (box_y2-offsetY):
        targetHitCounter += 1
        return True
    else:
        targetHitCounter = 0
        return False

def is_crosshair_locked(crosshair_x, crosshair_y, box_x1, box_y1, box_x2, box_y2):
    offsetX = (box_x2-box_x1)//2-((box_x2-box_x1)//12)
    offsetY = (box_y2-box_y1)//2-((box_y2-box_y1)//12)
    if (box_x1+offsetX) < crosshair_x//2 < (box_x2-offsetX) and (box_y1+offsetY) < crosshair_y//2 < (box_y2-offsetY):
        return True
    else:
        return False

# Check if crosshair is inside center of the box
def is_crosshair_inside_box(crosshair_x, crosshair_y, box_x1, box_y1, box_x2, box_y2):
    if box_x1 < crosshair_x//2 < box_x2 and box_y1 < crosshair_y//2 < box_y2:
        return True
    else:
        return False

def draw_faces(request):
    global firing
    global targetHitCounter
    global targetIndex
    global crosshairCol
    global thickness
    global freq
    global frame_rate_calc
    global t1
    global targets
    target = False
    onTarget = False
    GPIO.output(buzzer_pin, GPIO.LOW)
    t1 = cv2.getTickCount()
    with MappedArray(request, "lores") as m:
        input_shape = input_details[0]['shape']
        # If yuv is actually YUV420
        rgb = cv2.cvtColor(m.array, cv2.COLOR_YUV2BGR_I420)
        input_data = np.expand_dims(cv2.resize(rgb, (input_shape[1], input_shape[2])), axis=0)
        input_data = (input_data / 1.0).astype(np.uint8)  # <-- Cast to UINT8
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # Run inference
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index
        scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence scores
        percentage = 25
        with MappedArray(request, "main") as m2:
            filtered_boxes = []
            filtered_classes = []
            filtered_scores = []
            filtered_count = []
            for i in range(len(boxes)):
                if (classes[i] == 0) and int(scores[i]*100) >= percentage:
                    filtered_boxes.append(boxes[i])
                    filtered_classes.append(classes[i])
                    filtered_scores.append(scores[i])
            boxes = filtered_boxes
            classes = filtered_classes
            scores = filtered_scores
            count = len(classes)
            boxes, classes, scores = non_maximum_suppression(boxes, classes, scores, 0.5, percentage/100)
            targets = boxes
            text = "pew pew"
            if(firing):
                text="Beserker"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 5
            font_thickness = 10
            color = (255, 0, 0)  # White color
            # Get the width of the text
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            # Calculate the starting X coordinate to center the text
            start_x = (m2.array.shape[1] - text_width) // 2
            # Define the starting Y coordinate (you can adjust this value to position the text higher or lower)
            start_y = 120  # 30 pixels from the top
            if(1):
                for i, score in enumerate(scores):
                    ymin, xmin, ymax, xmax = boxes[i]
                    x1, y1, x2, y2 = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)
                    if(is_crosshair_in_center(crosshairX, crosshairY, x1, y1, x2, y2) or firing):
                        onTarget=True
                        #if(not firing):
                        offsetX = (x2-x1)//2-((x2-x1)//4)
                        offsetY = (y2-y1)//2-((y2-y1)//4)
                        cv2.rectangle(m2.array, (int(x1)+offsetX, int(y1)+offsetY), (int(x2)-offsetX, int(y2)-offsetY), (255, 0, 0), 25)
                        cv2.rectangle(m2.array, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 30)
                        # Put the text on the image
                        cv2.putText(m2.array, text, (start_x, start_y), font, font_scale, color, font_thickness)
                    elif(is_crosshair_inside_box(crosshairX, crosshairY, x1, y1, x2, y2)):
                        target = True
                        cv2.rectangle(m2.array, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 30)
                    else:
                        cv2.rectangle(m2.array, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 15)

                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(y1, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(m2.array, (x1, label_ymin-labelSize[1]-10), (x1+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(m2.array, label, (x1, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            if target:
                crosshairCol = (255, 255, 0)
            elif(onTarget or firing):
                GPIO.output(buzzer_pin, GPIO.HIGH)
                crosshairCol = (255, 0, 0)
            else:
                crosshairCol = (0, 255, 0)
            cv2.putText(m2.array, 'FPS: {}'.format( math.ceil(frame_rate_calc)), (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(m2.array, 'Targets: {}'.format(len(targets)), (30,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(m2.array, 'Pan: {}'.format(str(pan)), (30,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(m2.array, 'Tilt: {}'.format(str(tilt)), (30,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(m2.array, 'Hits: {}'.format(int(targetHitCounter)), (30,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(m2.array, 'Counter: {}'.format(int(counter)), (30,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA)

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (width, height)},
                                             lores={"size": (original_width, original_height), "format": "YUV420"})
config["transform"] = libcamera.Transform(hflip=1, vflip=1)
picam2.options["quality"] = 100
picam2.configure(config)
picam2.start_preview(Preview.QTGL, x=0, y=0, width=width, height=height)

# Get the current date and time
now = datetime.datetime.now()

# Format the date and time as a string
filename = now.strftime("%m-%d_%H-%M-%S.mp4")

encoder = H264Encoder(10000000)
output = FfmpegOutput(filename)


picam2.start_and_record_video(output, encoder, show_preview=True)

FRAME_RATE = 30
FRAME_RATE = 1000000 // FRAME_RATE
picam2.set_controls({"FrameDurationLimits":(FRAME_RATE,FRAME_RATE)})

picam2.start()
picam2.set_controls({"AfMode": 2 ,"AfTrigger": 0})
targets = []
picam2.post_callback = draw_faces

def draw_crosshair(request):
    global pan
    global tilt
    global firing
    global targetHitCounter
    global targetIndex
    global crosshairX
    global crosshairY
    global counter
    global crosshairCol
    global t3
    global freq2
    global targets
    global frame_rate_calc2
    counter += 1

    if(len(targets)==0 and counter%50 == 0):
        firing = 0
        GPIO.output(buzzer_pin, GPIO.LOW)
    if(len(targets)==1 and counter%3 == 0):
        firing = 0
        y1, x1, y2, x2 = targets[0]
        x1 = x1 * width
        y1 = y1 * height
        x2 = x2 * width
        y2 = y2 * height
        center_x = x2 - x1
        object_center_x = (x1 + x2) / 2
        object_center_y = (y1 + y2) / 2
        error_x = width//2 - object_center_x
        error_y = height//2 - object_center_y
# Inverted adjustments using only P-control
        adjustment_x = -kp * error_x
        adjustment_y = kp * error_y
# Limit the adjustments to a maximum value
        max_adjustment = 20  # You can adjust this value
        adjustment_x = -max(-max_adjustment, min(max_adjustment, adjustment_x))
        adjustment_y = -max(-max_adjustment, min(max_adjustment, adjustment_y))
        pan = int(pan + (adjustment_x))
        tilt = int(tilt + adjustment_y)
# Ensure the new values are within the range
        pan = max(-90, min(90, pan))
        tilt = max(-90, min(30, tilt))
# Set the constrained values
        if(is_crosshair_locked(crosshairX, crosshairY,x1,y1,x2,y2)):
            pass
        else:
            panServo.set_angle(pan)
            tiltServo.set_angle(tilt)

    if(len(targets)>=2 and counter%1 == 0):
        if(len(targets)>2):
            firing = 1
        if(counter%2 == 0):
            targetIndex = get_leftmost_object_index(targets)
        elif(counter%2 == 1):
            targetIndex = get_rightmost_object_index(targets)
        try:
            y1, x1, y2, x2 = targets[targetIndex]
            x1 = x1 * width
            y1 = y1 * height
            x2 = x2 * width
            y2 = y2 * height
            center_x = x2 - x1
            object_center_x = (x1 + x2) / 2
            object_center_y = (y1 + y2) / 2
            error_x = width//2 - object_center_x
            error_y = height//2 - object_center_y
# Inverted adjustments using only P-control
            adjustment_x = -kp*2 * error_x
            adjustment_y = kp * error_y
# Limit the adjustments to a maximum value
            max_adjustment = 90  # You can adjust this value
            adjustment_x = -max(-max_adjustment, min(max_adjustment, adjustment_x))
            adjustment_y = -max(-max_adjustment, min(max_adjustment, adjustment_y))
            print("\nObject Center X:", object_center_x)
            print("Object Center Y:", object_center_y)
            print("Error X:", error_x)
            print("Error Y:", error_y)
            print("adjustment_x: "+str(adjustment_x)) # = kp * error_x
            print("adjustment_y: "+str(adjustment_y)) #= kp * error_y
            pan = int(pan + (adjustment_x))
            tilt = int(tilt + (adjustment_y))
# Ensure the new values are within the range
            pan = max(-90, min(90, pan))
            tilt = max(-90, min(30, tilt))
            panServo.set_angle(pan)
            tiltServo.set_angle(tilt)
        except Exception as e:
            targetIndex = 0
            print(str(e))
    with MappedArray(request, "main") as m:
        new_frame_time = time.time()
        # Calculating the fps
        t3 = cv2.getTickCount()
        #Draw crosshair
        cv2.line(m.array, (crosshairX//2 - crosshairW, crosshairY//2), (crosshairX//2 + crosshairW, crosshairY//2), crosshairCol, 2)  # Horizontal line
        cv2.line(m.array, (crosshairX//2, crosshairY//2 - crosshairW), (crosshairX//2, crosshairY//2 + crosshairW), crosshairCol, 2)  # Vertical line

        if(frame_rate_calc2 < 100):
            cv2.putText(m.array,'FPS: {}'.format(int(frame_rate_calc2)),(30,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        else:
            cv2.putText(m.array,'FPS:{}'.format(int(frame_rate_calc2)),(30,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

        # Calculate framerate
        t4 = cv2.getTickCount()
        time2 = (t4-t3)/freq2
        frame_rate_calc2= 1/time2

picam2.pre_callback = draw_crosshair

size = picam2.capture_metadata()['ScalerCrop'][2:]
print("SS: "+str(size))
full_res = picam2.camera_properties['PixelArraySize']

current_settings = picam2.capture_metadata()['ScalerCrop']
current_offset = current_settings[:2]
current_size = current_settings[2:]
print("CO: "+str(current_offset))
print("CS: "+str(current_size))

start_time = time.monotonic()
# Run for 100 seconds to display the camera feed.

pan_increment = 1  # Angle increment for each step
max_angle = 90     # Maximum angle to reach
min_angle = -90    # Minimum angle to reach
scan_delay = 0.025  # Delay between each pan step (adjust as needed)

try:
    while time.monotonic() - start_time < 300:
        # Pan to the left (negative direction)
        while pan > min_angle:
            if(len(targets) != 0):
                break
            panServo.set_angle(pan)
            pan -= pan_increment
            time.sleep(scan_delay)
        while(len(targets) != 0):
        # Pause for a moment at the leftmost position
            time.sleep(2)
        # Pan back to the right (positive direction)
        while pan < max_angle:
            if(len(targets) != 0):
                break
            panServo.set_angle(pan)
            pan += pan_increment
            time.sleep(scan_delay)
        while(len(targets) != 0):
            time.sleep(2)
except KeyboardInterrupt:
    # Clean up and release the GPIO pin
    #GPIO.cleanup()
    print("\nKeyboard interrupt detected! Exiting gracefully...")
    # Any cleanup code or other tasks you want to run before exiting can be placed here
finally:
    print("Program has exited.")
    # Clean up and release the GPIO pin
    GPIO.cleanup()