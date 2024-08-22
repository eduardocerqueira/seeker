#date: 2024-08-22T16:51:03Z
#url: https://api.github.com/gists/68cc611203d9cc16917f3b5d6376a199
#owner: https://api.github.com/users/raspiduino

# Import
import cv2
from deepsparse.pipeline import Pipeline
from deepsparse.yolo.schemas import YOLOInput
from deepsparse.yolo.utils import COCO_CLASSES
import time

# Model settings
task = "yolo"
model_path = "zoo:yolov5-l-coco-pruned.4block_quantized"
iou_thres=0.25
conf_thres=0.45

# Create pipeline
pipeline = Pipeline.create(task, model_path=model_path, batch_size=1)

def get_color(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] * 
    (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

def process_frame(frame):
    # Convert to YOLO input format
    yolo_input = YOLOInput(iou_thres=iou_thres, conf_thres=conf_thres, images=frame)

    # Process frame
    yolo_output = pipeline(yolo_input)

    # Parse result
    boxes = yolo_output.boxes[0]
    labels = yolo_output.labels[0]
    scores = yolo_output.scores[0]

    # Draw boxes
    for i in range(len(boxes)):
        # Get box
        box = boxes[i]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        # Get label
        label = int(labels[i])
        color = get_color(label)

        # Get score
        score = scores[i]

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1, x2, y2), color, 2)

        # Draw class name and confidence
        cv2.putText(frame, f'{COCO_CLASSES[label]} {score:.2f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame

if __name__ == "__main__":
    # Init camera
    cam = cv2.VideoCapture(0)

    # Main loop
    while True:
        # Get frame
        ret, frame = cam.read()
        if not ret:
            continue

        # Process frame and draw bounding boxes
        time1 = time.time()
        frame = process_frame(frame)
        time2 = time.time()

        # Draw fps
        cv2.putText(frame, f'{1/(time2 - time1):.1f} FPS', (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow('frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and destroy all windows
    cam.release()
    cv2.destroyAllWindows()
