#date: 2024-04-03T16:59:33Z
#url: https://api.github.com/gists/62d1f8f60268103694a44573895c34f5
#owner: https://api.github.com/users/travelwor1d

# Install memory-profiler using pip:
# pip install memory-profiler

import cv2
import numpy as np
from memory_profiler import profile

@profile
def process_image(image_path):
    # Read image
    img = cv2.imread(image_path)

    # Perform image processing operations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Display processed image (for demonstration)
    cv2.imshow("Processed Image", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "example.jpg"
    process_image(image_path)
