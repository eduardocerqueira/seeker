#date: 2023-02-20T16:50:02Z
#url: https://api.github.com/gists/f0ebe7084f9c21aa8d267e19ddf37d7e
#owner: https://api.github.com/users/Lamroy95

import sys

import cv2
from PyQt6.QtWidgets import QApplication

from opencv_gui_wrapper import Range, CvGuiWrapper


CANNY_PARAMS = {
    "lower_thresh": Range(0, 100, 5, 30),
    "high_thresh": Range(0, 100, 5, 60)
}

BILATERAL_FILTER_PARAMS = {
    "d": Range(1, 10, 1, 5),
    "sigmaColor": Range(0, 300, 5, 50),
    "sigmaSpace": Range(0, 300, 5, 50),
}


def main():
    app = QApplication(sys.argv)

    img = cv2.imread("smoge.jpg")
    CvGuiWrapper(cv2.Canny, img, CANNY_PARAMS)
    CvGuiWrapper(cv2.bilateralFilter, img, BILATERAL_FILTER_PARAMS)

    app.exec()


if __name__ == "__main__":
    main()