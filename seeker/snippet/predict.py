#date: 2024-10-29T16:49:54Z
#url: https://api.github.com/gists/bfdbb319c05025faec7fda56bba839ab
#owner: https://api.github.com/users/RizwanMunawar

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
results = model("path/to/video/file.mp4")  # predict on a video
