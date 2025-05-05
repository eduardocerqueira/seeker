#date: 2025-05-05T16:49:42Z
#url: https://api.github.com/gists/d8582c69e6115827d861c20a48522f63
#owner: https://api.github.com/users/Bhuvan0811

import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """ Calculate angle between three points """
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def classify_pose(landmarks, image_height):
    # Get coordinates
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

    # Calculate midpoints
    mid_hip = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
    mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
    mid_knee = [(left_knee[0] + right_knee[0]) / 2, (left_knee[1] + right_knee[1]) / 2]

    # Calculate angles
    hip_angle = calculate_angle(mid_shoulder, mid_hip, mid_knee)

    # Get body vertical span (distance from shoulders to hips)
    vertical_span = abs(mid_shoulder[1] - mid_hip[1]) * image_height

    # Basic rules
    if vertical_span < image_height * 0.1:
        return "Lying"
    elif 70 < hip_angle < 110:
        return "Sitting"
    else:
        return "Standing"

# ------------- MAIN ----------------

# Load your image
image_path = 'standing.jpg'  # <<< CHANGE to your image path
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image
results = pose.process(image_rgb)

if results.pose_landmarks:
    # Draw pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Classify pose
    label = classify_pose(results.pose_landmarks.landmark, image_height)
    print(f"Detected Pose: {label}")

    # Put label on image
    cv2.putText(image, label, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
else:
    print("No person detected!")

# Show the output
cv2.imshow('Pose Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
