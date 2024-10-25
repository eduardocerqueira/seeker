#date: 2024-10-25T17:11:05Z
#url: https://api.github.com/gists/3f9d9e623dad352288e0fb930b6afadd
#owner: https://api.github.com/users/anandsuraj

import os
import rawpy
from PIL import Image

#output_quality
#  90–100: Excellent quality, very minimal compression (large file sizes).
# 80–90: High quality, slight compression with minimal visible loss (recommended for web images).
# 60–80: Moderate quality, noticeable compression, smaller file sizes (good for thumbnails).
# 30–60: Low quality, higher compression, visible artifacts (suitable for previews).
# 1–30: Very low quality, maximum compression (not recommended unless size is a top priority).

def optimize_image(image_path, output_quality=85):
    """Optimizes an image and saves it with good quality settings."""
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")  # Ensure compatibility
        img.save(image_path, "JPEG", quality=output_quality, optimize=True)
        print(f"Optimized: {image_path}")
    except Exception as e:
        print(f"Could not optimize {image_path}: {e}")

def process_cr2_image(cr2_path, output_quality=85):
    """Processes CR2 files and converts them to optimized JPEGs."""
    try:
        with rawpy.imread(cr2_path) as raw:
            rgb = raw.postprocess()
            img = Image.fromarray(rgb)
            img = img.convert("RGB")
            output_path = os.path.splitext(cr2_path)[0] + ".jpg"  # Save as JPEG
            img.save(output_path, "JPEG", quality=output_quality, optimize=True)
            print(f"Converted and optimized: {output_path}")
            os.remove(cr2_path)  # Optionally delete the original CR2 file
    except Exception as e:
        print(f"Could not process {cr2_path}: {e}")

def optimize_images_in_directory(directory, output_quality=85):
    """Walks through a directory and optimizes all supported image files."""
    supported_formats = (".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif", ".cr2")

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Process CR2 files
            if file.lower().endswith(".cr2"):
                process_cr2_image(file_path, output_quality)
            # Process other image formats
            elif file.lower().endswith(supported_formats):
                optimize_image(file_path, output_quality)

# Provide the main directory path here
main_directory = "/Users/"
optimize_images_in_directory(main_directory, output_quality=85)