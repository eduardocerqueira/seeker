#date: 2024-10-29T16:54:35Z
#url: https://api.github.com/gists/22dcc47df2f79ca7ba4b7f5fa4514460
#owner: https://api.github.com/users/RizwanMunawar

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-obb.pt")  # load an official model
model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model("https://ultralytics.com/images/boats.jpg")  # predict on an image
results = model("Path/to/video/file.mp4")  # predict on a video
