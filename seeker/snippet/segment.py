#date: 2024-10-29T16:51:42Z
#url: https://api.github.com/gists/ff6d58ea6e9b2193f302101d88f7c7c6
#owner: https://api.github.com/users/RizwanMunawar

import cv2  # OpenCV library for image/video processing
from ultralytics import YOLO  # Ultralytics YOLO model for object detection/segmentation
from ultralytics.utils.plotting import Annotator, colors  # Utilities for annotating and visualizing YOLO results

# Load the Ultralytics YOLO11 segmentation model
model = YOLO("yolo11n-seg.pt")

# Get the class names from the model
names = model.model.names

# Open the video capture
cap = cv2.VideoCapture("path/to/video/file.mp4")

while True:
    # Read a frame from the video
    ret, im0 = cap.read()
    if not ret:
        break  # Break the loop if we reach the end of the video

    # Run the YOLO model on the frame
    results = model.predict(im0)

    # Create an annotator to draw bounding boxes and masks
    annotator = Annotator(im0, line_width=2)

    # If the model detected any objects with masks
    if results[0].masks is not None:
        # Get the class IDs and mask coordinates
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy

        # Iterate over the detected objects
        for mask, cls in zip(masks, clss):
            # Get the color and text color for the current class
            color = colors(int(cls), True)
            txt_color = annotator.get_txt_color(color)

            # Annotate the frame with the segmented object and its class label
            annotator.seg_bbox(mask=mask, mask_color=color, label=names[int(cls)], txt_color=txt_color)

    # Write the annotated frame to the output video
    out.write(im0)

    # Display the annotated frame
    cv2.imshow("instance-segmentation", im0)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and output writer
out.release()
cap.release()
cv2.destroyAllWindows()