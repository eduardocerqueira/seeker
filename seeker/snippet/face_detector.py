#date: 2025-11-27T16:48:42Z
#url: https://api.github.com/gists/c2bc9ad9e87dc748170233bf0fd16981
#owner: https://api.github.com/users/parsapoorsh

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class FaceDetection:
    face_xy1: Tuple[int, int]
    face_wh: Tuple[int, int]
    right_eye: Tuple[int, int]
    left_eye: Tuple[int, int]
    nose_tip: Tuple[int, int]
    mouth_right: Tuple[int, int]
    mouth_left: Tuple[int, int]
    confidence: float

    @property
    def face_xy2(self):
        fx, fy = self.face_xy1
        fw, fh = self.face_wh
        return (fx + fw,
                fy + fh)

    @property
    def face_center(self):
        fx, fy = self.face_xy1
        fw, fh = self.face_wh
        return (fx + (fw // 2),
                fy + (fh // 2))


class FaceDetector:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path).absolute().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)

        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.7,
            nms_threshold=0.4,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

    def detect(self, img: np.ndarray) -> Tuple[FaceDetection]:
        iw, ih, _ = img.shape
        self.detector.setInputSize((ih, iw))
        __, faces = self.detector.detect(img)
        if faces is None:
            faces = []

        result = tuple(
            FaceDetection(
                face_xy1=tuple(map(int, f[:2])),
                face_wh=tuple(map(int, f[2:4])),
                right_eye=tuple(map(int, f[4:6])),
                left_eye=tuple(map(int, f[6:8])),
                nose_tip=tuple(map(int, f[8:10])),
                mouth_right=tuple(map(int, f[10:12])),
                mouth_left=tuple(map(int, f[12:14])),
                confidence=float(f[14]),
            )
            for f in faces
        )
        return result

    @staticmethod
    def visualize(
        img: np.ndarray,
        faces: Tuple[FaceDetection],
        radius: int = 4,
        thickness: int = 4,
    ) -> np.ndarray:
        BRG_BLUE = (255, 0, 0)
        BRG_RED = (0, 0, 255)
        BRG_GREEN = (0, 255, 0)
        BRG_CYAN = (255, 255, 0)
        BRG_YELLOW = (0, 255, 255)

        vimg = img.copy()
        for f in faces:
            cv2.rectangle(vimg, f.face_xy1, f.face_xy2, BRG_GREEN, thickness)
            cv2.circle(vimg, f.face_center, radius, BRG_CYAN, thickness)
            cv2.circle(vimg, f.right_eye, radius, BRG_RED, thickness)
            cv2.circle(vimg, f.left_eye, radius, BRG_RED, thickness)
            cv2.circle(vimg, f.nose_tip, radius, BRG_YELLOW, thickness)
            cv2.line(vimg, f.mouth_left, f.mouth_right, BRG_BLUE, thickness)

            cv2.putText(
                vimg,
                f"{f.confidence:.2f}",
                (f.face_xy1[0], f.face_xy1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                BRG_GREEN,
                thickness,
            )
        return vimg


if __name__ == "__main__":
    # download the model from:
    # https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

    input_img_path = Path("test.jpg")
    output_image_path = input_img_path.absolute().with_stem(input_img_path.stem + "-visualized_faces")
    model_path = Path("face_detection_yunet_2023mar.onnx")

    fd = FaceDetector(model_path=model_path)

    if not input_img_path.is_file():
        raise FileNotFoundError(input_img_path)
    img = cv2.imread(input_img_path)

    faces = fd.detect(img)
    print(f"Found {len(faces)} face(s)")

    vimg = fd.visualize(img, faces)
    cv2.imwrite(
        filename=output_image_path,
        img=vimg,
    )
