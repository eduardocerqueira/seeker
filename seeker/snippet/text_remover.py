#date: 2025-04-24T16:48:29Z
#url: https://api.github.com/gists/91c97de01016af8e09ae96c4ae894756
#owner: https://api.github.com/users/abuibrahimjega

import cv2
import easyocr
import numpy as np

# Load image
image_path = "2.JPG"
img = cv2.imread(image_path)

# Initialize EasyOCR Reader (English only, add more languages if needed)
reader = easyocr.Reader(['en'], gpu=False)

# Run OCR
results = reader.readtext(img)

# Create mask
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# Loop through all detected texts
for (bbox, text, prob) in results:
    # Unpack the bounding box
    (top_left, top_right, bottom_right, bottom_left) = bbox
    pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)  # Fill detected text area on mask

# Inpaint to remove text
inpainted = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Save the result
cv2.imwrite("cleaned_easyocr.jpg", inpainted)
print("✅ Saved: cleaned_easyocr.jpg")