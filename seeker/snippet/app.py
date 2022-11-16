#date: 2022-11-16T17:00:52Z
#url: https://api.github.com/gists/66bf3c9b74abb0aec46016f10df30739
#owner: https://api.github.com/users/kaysush

from flask import Flask,render_template,Response
from unittest import result
import numpy as np
import cv2
import time 
import tensorflow as tf
import tensorflow_hub as hub
from matplotlib import pyplot as plt
import win32api
import pyttsx3
import pythoncom
from time import sleep
import schedule
import time
import matplotlib.pyplot as plt
import gtts  
from playsound import playsound  
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


app=Flask(__name__)



def gen_frames():  
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = pose.process(image)

            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                 landmarks = results.pose_landmarks.landmark
                 l_elbow=[landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                 r_elbow=[landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                 l_knee=[landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                 r_knee=[landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                                  
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route("/")
def home():
     return render_template("index.html")

@app.route("/feed/")
def feed():
     return render_template("Feedback.html")

@app.route("/start/")
def start():
     return render_template("start.html")

@app.route("/start1/")
def start1():
     return render_template("start1.html")

@app.route("/start2/")
def start2():
     return render_template("start2.html")

@app.route("/start3/")
def start3():
     return render_template("start3.html")

@app.route("/vid1/")
def vid1():
     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/vid2/")
def vid2():
     return Response()

@app.route("/vid3/")
def vid3():
     return Response()

@app.route("/vid4/")
def vid4():
     return Response()


if __name__=="__main__":
    app.run(host = "127.0.0.1",debug=True)