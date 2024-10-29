#date: 2024-10-29T16:53:05Z
#url: https://api.github.com/gists/f84d034142f21fd837313f5ab66c1b2d
#owner: https://api.github.com/users/RizwanMunawar

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")  # load an official model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
results = model("Path/to/video/file.mp4")  # predict on a video