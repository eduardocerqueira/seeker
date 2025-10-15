#date: 2025-10-15T16:53:59Z
#url: https://api.github.com/gists/d402040329388a71b5e7673a01d70e8d
#owner: https://api.github.com/users/warning-machines

#!/usr/bin/env python3
# Copyright 2023-2024 NXP
# SPDX-License-Identifier: MIT

import cv2
import tensorflow as tf
import ai_edge_litert.interpreter as lite
import numpy as np
import time
import random
import inspect
import serial

arduino_serial = serial.Serial('/dev/ttymxc2', baudrate=9600, timeout=0)

random.seed(42)

DEFAULT_OBJECT_DETECTOR_TFLITE = "yolov4-tiny_416_quant.tflite"
DEFAULT_IMAGE_FILENAME = "WIN_20250325_12_55_46_Pro.jpg"

CAM_FOV = 55 #degrees

COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite" "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

SCORE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
INFERENCE_IMG_SIZE = 416
MAX_DETS = 1

ANCHORS = [[[81, 82], [135, 169], [344, 319]], [[23, 27], [37, 58], [81, 82]]]
SIGMOID_FACTOR = [1.05, 1.05]
NUM_ANCHORS = 3
STRIDES = [32, 16]
GRID_SIZES = [int(INFERENCE_IMG_SIZE / s) for s in STRIDES]


def load_interpreter(args):

    ext_delegate = None
    ext_delegate_options = {}

    # parse extenal delegate options
    if args.ext_delegate_options is not None:
        options = args.ext_delegate_options.split(";")
        for o in options:
            kv = o.split(":")
            if len(kv) == 2:
                ext_delegate_options[kv[0].strip()] = kv[1].strip()
            else:
                raise RuntimeError("Error parsing delegate option: " + o)

    # load external delegate
    if args.ext_delegate is not None:
        print(
            "Loading external delegate from {} with args: {}".format(
                args.ext_delegate, ext_delegate_options
            )
        )
        ext_delegate = [lite.load_delegate(args.ext_delegate, ext_delegate_options)]

    interpreter = lite.Interpreter(
        model_path=args.model_file,
        experimental_delegates=ext_delegate,
        num_threads=args.num_threads,
    )
    interpreter.allocate_tensors()
    return interpreter


def gen_box_colors():
    colors = []
    for _ in range(len(COCO_CLASSES)):
        r = random.randint(100, 255)
        g = random.randint(100, 255)
        b = random.randint(100, 255)
        colors.append((r, g, b))

    return colors


BOX_COLORS = gen_box_colors()


def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (INFERENCE_IMG_SIZE, INFERENCE_IMG_SIZE))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))


def reciprocal_sigmoid(x):
    return -np.log(1 / x - 1)


def decode_boxes_prediction(yolo_output):
    # Each output level represents a grid of predictions.
    # The first output level is a 26x26 grid and the second 13x13.
    # Each cell of each grid is assigned to 3 anchor bounding boxes.
    # The bounding box predictions are regressed
    # relatively to these anchor boxes.
    # Thus, the model predicts 3 bounding boxes per cell per output level.
    # The output is structured as follows:
    # For each cell [[x, y, w, h, conf, cl_0, cl_1, ..., cl_79], # anchor 1
    #                [x, y, w, h, conf, cl_0, cl_1, ..., cl_79], # anchor 2
    #                [x, y, w, h, conf, cl_0, cl_1, ..., cl_79]] # anchor 3
    # Hence, we have 85 values per anchor box, and thus 255 values per cell.
    # The decoding of the output bounding boxes is described in Figure 2 of
    # the YOLOv3 paper https://arxiv.org/pdf/1804.02767.pdf;

    boxes_list = []
    scores_list = []
    classes_list = []

    for idx, feats in enumerate(yolo_output):

        features = np.reshape(feats, (NUM_ANCHORS * GRID_SIZES[idx] ** 2, 85))

        anchor = np.array(ANCHORS[idx])
        factor = SIGMOID_FACTOR[idx]
        grid_size = GRID_SIZES[idx]
        stride = STRIDES[idx]

        cell_confidence = features[..., 4]
        logit_threshold = reciprocal_sigmoid(SCORE_THRESHOLD)
        over_threshold_list = np.where(cell_confidence > logit_threshold)

        if over_threshold_list[0].size > 0:
            indices = np.array(over_threshold_list[0])

            box_positions = np.floor_divide(indices, 3)

            list_xy = np.array(np.divmod(box_positions, grid_size)).T
            list_xy = list_xy[..., ::-1]
            boxes_xy = np.reshape(list_xy, (int(list_xy.size / 2), 2))

            outxy = features[indices, :2]

            # boxes center coordinates
            centers = np_sigmoid(outxy * factor) - 0.5 * (factor - 1)
            centers += boxes_xy
            centers *= stride

            # boxes width and height
            width_height = np.exp(features[indices, 2:4])
            width_height *= anchor[np.divmod(indices, NUM_ANCHORS)[1]]

            boxes_list.append(
                np.stack(
                    [
                        centers[:, 0] - width_height[:, 0] / 2,
                        centers[:, 1] - width_height[:, 1] / 2,
                        centers[:, 0] + width_height[:, 0] / 2,
                        centers[:, 1] + width_height[:, 1] / 2,
                    ],
                    axis=1,
                )
            )

            # confidence that cell contains an object
            scores_list.append(np_sigmoid(features[indices, 4:5]))

            # class with the highest probability in this cell
            classes_list.append(np.argmax(features[indices, 5:], axis=1))

    if len(boxes_list) > 0:
        boxes = np.concatenate(boxes_list, axis=0)
        scores = np.concatenate(scores_list, axis=0)[:, 0]
        classes = np.concatenate(classes_list, axis=0)

        return boxes, scores, classes
    else:
        return np.zeros((0, 4)), np.zeros((0)), np.zeros((0))


def decode_output(
    yolo_outputs, score_threshold=SCORE_THRESHOLD, iou_threshold=NMS_IOU_THRESHOLD
):
    """
    Decode output from YOLOv4 tiny in inference size referential (416x416)
    """
    boxes, scores, classes = decode_boxes_prediction(yolo_outputs)

    # apply NMS from tensorflow
    inds = tf.image.non_max_suppression(
        boxes,
        scores,
        MAX_DETS,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )

    # keep only selected boxes
    boxes = tf.gather(boxes, inds)
    scores = tf.gather(scores, inds)
    classes = tf.gather(classes, inds)

    return scores, boxes, classes


def run_inference(interpreter, image, threshold=SCORE_THRESHOLD):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale, input_zero_point = input_details[0]["quantization"]
    image = image / input_scale + input_zero_point
    image = image.astype(np.int8)

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]["index"])
    boxes2 = interpreter.get_tensor(output_details[1]["index"])

    return [boxes, boxes2]

def video_process(args):

    try:
        cap = cv2.VideoCapture(args.source)
    except:
        raise("Invalid data input. Try with /dev/videoX or a webcam feed streaming url")
    
    start = time.perf_counter()
    interpreter = load_interpreter(args=args)
    print("Interpreter loading time", (time.perf_counter() - start) * 1000, "ms")

    arduino_serial.write("TEST\n".encode('utf-8'))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_image = preprocess_image(frame)

        start = time.perf_counter()
        yolo_output = run_inference(interpreter, processed_image)
        end = time.perf_counter()

        scores, boxes, classes = decode_output(yolo_output)

        # rescale boxes for display
        shp = frame.shape
        boxes = boxes.numpy()
        boxes /= INFERENCE_IMG_SIZE
        boxes *= np.array([shp[1], shp[0], shp[1], shp[0]])

        boxes = boxes.astype(np.int32)

        if first_inference:
            print("First inference time", end - start, "ms")
            first_inference = False
        else:
            print("Inference time", end - start, "ms")

        # print("Detected", boxes.shape[0], "object(s)")
        for i in range(boxes.shape[0]):
            if classes[i].numpy() != 0:
                continue

            box = boxes[i, :]
            box_x_center = int((box[0] + box[2])/2)
            box_y_center = int((box[1] + box[3])/2)
            box_w = int(box[2] - box[0])
            box_h = int(box[3] - box[1])
            print([box_x_center, box_y_center, box_w, box_h], end=" ")
            class_name = COCO_CLASSES[classes[i].numpy()]
            score = scores[i].numpy()
            print("class", class_name, end=" ")
            print("score", score)

            if hasattr(cv2, "imshow") and inspect.isfunction(cv2.imshow):
                color = BOX_COLORS[classes[i]]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 3)
                cv2.putText(
                    frame,
                    f"{class_name} {score:.2f}",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
            
            pan_delta_angle = int(CAM_FOV*(box_x_center - frame.shape[0]//2)/frame.shape[0])
            print("\n")
            print(f"Angle sent : {pan_delta_angle}")
            message = f"P:{pan_delta_angle},T:{0}\n"
            arduino_serial.flush()
            arduino_serial.write(message.encode('utf-8'))
            time.sleep(0.1)

        if hasattr(cv2, "imshow") and inspect.isfunction(cv2.imshow):
            cv2.imwrite("example_output.jpg", frame)
            cv2.imshow("", frame)
