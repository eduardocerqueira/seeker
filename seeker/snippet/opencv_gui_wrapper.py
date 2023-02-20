#date: 2023-02-20T16:50:02Z
#url: https://api.github.com/gists/f0ebe7084f9c21aa8d267e19ddf37d7e
#owner: https://api.github.com/users/Lamroy95

from abc import ABC
from dataclasses import dataclass
from functools import partial
from typing import Optional, Iterable

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QSlider, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
)


class ParamWidget(ABC):
    @property
    def value(self):
        raise NotImplementedError

    def render(self, name: str) -> Iterable[QWidget]:
        raise NotImplementedError


@dataclass
class Range(ParamWidget):
    start: int
    stop: int
    step: int
    default: int = 0
    __qt_widget: Optional[QSlider] = None

    @property
    def value(self):
        return self.__qt_widget.value()

    def render(self, name: str) -> (QLabel, QSlider, QLabel):
        slider = QSlider(Qt.Orientation.Horizontal)
        self.__qt_widget = slider
        slider.setMinimum(self.start)
        slider.setMaximum(self.stop)
        slider.setSingleStep(self.step)
        slider.setValue(self.default)
        value_label = QLabel(str(self.default))
        slider.valueChanged.connect(value_label.setNum)
        return QLabel(name), slider, value_label


def cv_img_to_pixmap(cv_image) -> QPixmap:
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    height, width, *_ = cv_image.shape
    bpl = 3 * width
    qt_img = QImage(cv_image.data, width, height, bpl, QImage.Format.Format_RGB888)
    return QPixmap(qt_img)


class CvGuiWrapper(QWidget):
    def __init__(self, func, img, params: dict[str, ParamWidget]):
        super().__init__()
        self.params = params
        self.controls = QWidget()
        self.output = QWidget()
        self.result = QLabel()
        self.result.setPixmap(cv_img_to_pixmap(img))

        tab_layout = QHBoxLayout()
        controls_layout = QVBoxLayout()
        output_layout = QVBoxLayout()

        output_layout.addWidget(self.result)
        self.output.setLayout(output_layout)

        for param_name, type_ in params.items():
            row = QWidget()
            row_layout = QHBoxLayout()
            widgets = type_.render(param_name)
            for w in widgets:
                row_layout.addWidget(w)
            row.setLayout(row_layout)
            controls_layout.addWidget(row)

        self.func_btn = QPushButton("Execute")
        self.func_btn.clicked.connect(partial(self.executor_callback, func, img))
        controls_layout.addWidget(self.func_btn)

        self.controls.setLayout(controls_layout)
        tab_layout.addWidget(self.controls)
        tab_layout.addWidget(self.output)

        self.setLayout(tab_layout)
        self.setWindowTitle(func.__name__)

        self.show()

    def executor_callback(self, func, img):
        args = [p.value for p in self.params.values()]
        cv_image = func(img, *args)
        self.result.setPixmap(cv_img_to_pixmap(cv_image))


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
