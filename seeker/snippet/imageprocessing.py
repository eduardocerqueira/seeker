#date: 2023-04-13T16:51:43Z
#url: https://api.github.com/gists/a3ae08316066594aedb952b926fc82cf
#owner: https://api.github.com/users/Abdullah7175

import mediapipe as mp
import cv2
from typing import NamedTuple

# Define the body parts to be measured
body_parts = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}

image = cv2.imread("C:/Users/Abdullah Anis/Desktop/test5.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mp_pose = mp.solutions.pose

with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:

    # Process the image
    results = pose.process(image)

    # Get the landmarks for each body part
    landmarks = results.pose_landmarks.landmark

    # Get the height and width of the image
    height, width, _ = image.shape
    # Define the conversion rate
#    pixels_to_cm = 0.1

    # Calculate the body measurements
    measurements = {}
    for part, index in body_parts.items():
        if landmarks[index].visibility > 0.5:
            measurements[part] = (
                landmarks[index].x * width,
                landmarks[index].y * height,
            )

    # Print the body measurements
    print("Body Measurements:")
    print("-" *20)
#    for part, measurement in measurements.items():
#        print(f"{part}: {measurement[0]:.2f} x-axis")

    if "left_eye_outer" in measurements and "left_eye_inner" in measurements and "left_eye" in measurements:
        lefteye_size_pixels = (measurements["left_eye_outer"][0]) - (measurements["left_eye_inner"][0])
        print(f"Left EYE: {lefteye_size_pixels:.2f}")
    else:
        print("Could not calculate Left Eye size: eye landmarks not found")

    if "right_eye_inner" in measurements and "right_eye_outer" in measurements and "right_eye" in measurements:
        righteye_size_pixels = measurements["right_eye_inner"][0] - measurements["right_eye_outer"][0]
        print(f"Right EYE: {righteye_size_pixels:.2f}")
    else:
        print("Could not calculate Right eye size: eye landmarks not found")

    if "left_shoulder" in measurements and "right_shoulder" in measurements:
        shoulder_size_pixels = (measurements["left_shoulder"][0]) + (measurements["right_shoulder"][0])
        print(f"Shoulder size: {shoulder_size_pixels:.2f}")
    else:
        print("Could not calculate shoulder size: shoulder landmarks not found")

    if "left_hip" in measurements and "right_hip" in measurements:
        hip_size_pixels = measurements["left_hip"][0] - measurements["right_hip"][0]
        print(f"Waist size: {hip_size_pixels:.2f}")
    else:
        print("Could not calculate Waist size: Waist landmarks not found")