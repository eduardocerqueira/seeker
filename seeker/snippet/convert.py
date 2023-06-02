#date: 2023-06-02T16:59:25Z
#url: https://api.github.com/gists/244374edd55cc5ebb23fcafe6cd39f23
#owner: https://api.github.com/users/rizoadev

import cv2
from PIL import Image, ExifTags
import os


def convert_to_webp(image_path, output_path):
    img = Image.open(image_path)
    img.save(output_path, "webp")

    print(f"Image converted to WebP: {output_path}")


def clear_metadata(image_path, output_path):
    # Open the image
    img = Image.open(image_path)

    # Remove EXIF metadata
    data = list(img.getdata())
    clean_image = Image.new(img.mode, img.size)
    clean_image.putdata(data)

    # Save the image without metadata
    clean_image.save(output_path)

    print(f"Metadata cleared. Image saved to: {output_path}")


def auto_color_correction(image_path, clip_limit=0.5, tile_grid_size=(8, 8)):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)

    # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) on the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)

    # Merge the equalized L channel with the original A and B channels
    lab_eq = cv2.merge((l_eq, a, b))

    # Convert the LAB image back to BGR color space
    corrected_img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return corrected_img


def compress_image(input_path, output_path, quality=90):
    # Open the image
    img = Image.open(input_path)

    # Save the compressed image
    img.save(output_path, optimize=True, quality=quality)

    print(f"Image compressed and saved to: {output_path}")


# Path to the input image
image_path = "path_to_image.jpg"

# Perform automatic color correction
corrected_image = auto_color_correction(image_path)

# Save the corrected image to a file
output_path = "corrected_image.jpg"
cv2.imwrite(output_path, corrected_image)

print(f"Corrected image saved to: {output_path}")
clear_metadata("corrected_image.jpg", "corrected_image.jpg")
compress_image("corrected_image.jpg", "corrected_image2.jpg")
convert_to_webp("corrected_image2.jpg", "corrected_image.webp")
