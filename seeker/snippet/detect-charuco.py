#date: 2025-12-03T17:05:33Z
#url: https://api.github.com/gists/65880b58707e9399abad758ba75c3b58
#owner: https://api.github.com/users/robcowie

# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "opencv-python<4.13.0",
# ]
# ///

# uv run detect-charuco.py path/to/image

import argparse
import logging

import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_charuco_board(image_path, squares_x=10, squares_y=8):
    """Detect a charuco board in an image file.

    Args:
        image_path: Path to the image file.
        squares_x: Number of chessboard squares in X direction.
        squares_y: Number of chessboard squares in Y direction.

    Return:
        charuco_corners: Detected charuco corners.
        charuco_ids: IDs of detected corners.
        image: Annotated image with detections.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

    board = cv2.aruco.CharucoBoard(
        size=(squares_x, squares_y),
        squareLength=1.0,
        markerLength=0.75,
        dictionary=aruco_dict,
    )

    board.setLegacyPattern(True)

    # Detect charuco corners
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        charuco_detector.detectBoard(gray)
    )

    # Nothing detected
    if marker_ids is None or len(marker_ids) == 0:
        logger.info("No ArUco markers detected")
        return None, None, image

    # Markers detected
    logger.info(f"Detected {len(marker_ids)} ArUco markers")
    cv2.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

    # No corners detected
    if charuco_corners is None or len(charuco_corners) == 0:
        logger.info("No ChArUco corners detected")
        return None, None, image

    # Corners detected
    cv2.aruco.drawDetectedCornersCharuco(
        image, charuco_corners, charuco_ids, cornerColor=(0, 255, 0)
    )
    logger.info(f"Detected {len(charuco_corners)} ChArUco corners")

    return charuco_corners, charuco_ids, image


def main():
    parser = argparse.ArgumentParser(
        description="Detect a charuco board in an image file."
    )
    parser.add_argument(
        "image",
        help="Path to the image file containing the charuco board",
    )
    args = parser.parse_args()

    try:
        corners, ids, annotated_image = detect_charuco_board(args.image)

        # cv2.imshow("ChArUco Detection", annotated_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        cv2.imwrite("charuco_detected.jpg", annotated_image)
        logger.info("Saved annotated image to charuco_detected.jpg")

    except Exception as e:
        logger.exception(f"Error: {e}")


if __name__ == "__main__":
    main()
