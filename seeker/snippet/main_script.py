#date: 2025-10-15T16:54:40Z
#url: https://api.github.com/gists/2c03448ed0484597d9d899a524f6f74f
#owner: https://api.github.com/users/warning-machines

from vision import video_process

import argparse
from multiprocessing import Process

DEFAULT_OBJECT_DETECTOR_TFLITE = "yolov4-tiny_416_quant.tflite"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source",
        required=False,
        help="image or video/webcam flux/url flux to be processed",
    )
    parser.add_argument(
        "-m",
        "--model-file",
        default=DEFAULT_OBJECT_DETECTOR_TFLITE,
        help=".tflite model to be executed",
    )
    parser.add_argument(
        "--show-inference", action="store_true", help="Display video with detections"
    )
    parser.add_argument("--input_mean", default=127.5, type=float, help="input_mean")
    parser.add_argument(
        "--input_std", default=127.5, type=float, help="input standard deviation"
    )
    parser.add_argument(
        "--num_threads", default=None, type=int, help="number of threads"
    )
    parser.add_argument("-e", "--ext_delegate", help="external_delegate_library path")
    parser.add_argument(
        "-o",
        "--ext_delegate_options",
        help='external delegate options, \
            format: "option1: value1; option2: value2"',
    )

    args = parser.parse_args()

    # tilt_stepper = StepperMotor(("/dev/gpiochip1",1),("/dev/gpiochip1",2))

    p_vision = Process(target=video_process, args=(args, angle_queue))
    # p_stepper = Process(target=stepper_process, args=(tilt_stepper, angle_queue,))

    p_vision.start()
    # p_stepper.start()

    p_vision.join()
    # p_stepper.join()
