#date: 2025-11-17T17:06:24Z
#url: https://api.github.com/gists/3e4604d09c2238b44b087e0fb8fd5970
#owner: https://api.github.com/users/srihari-codes

# ============================================================================
#                              EXERCISE 1
#                       Image Transformations
# ============================================================================

from PIL import Image, ImageOps  # pip install Pillow
from skimage import io, transform  # pip install scikit-image

# Load the image
image = Image.open("image.jpg")

# 1. TRANSLATION (shift image)
values = (1, 0, -50, 0, 1, -30)  # shift left by 50 and up by 30
translated = image.transform(
    image.size,
    Image.Transform.AFFINE,
    values
)
translated.save("1_translated.jpg")

# 2. ROTATION
rotated = image.rotate(45)  # rotate 45 degrees
rotated.save("2_rotated.jpg")

# 3. REFLECTION (flip)
# Horizontal flip
reflected_h = ImageOps.mirror(image)
reflected_h.save("3_mirrored.jpg")

# Vertical flip
reflected_v = ImageOps.flip(image)
reflected_v.save("3_flipped.jpg")

# 4. SCALING
# Scale to 200% size
new_size = (
    int(image.width * 2),
    int(image.height * 2)
)
scaled_up = image.resize(new_size, Image.Resampling.BICUBIC)
scaled_up.save("4_scaled.jpg")

# 5. CROPPING
# Crop center 50% of image
left = image.width * 0.25
top = image.height * 0.25
right = image.width * 0.75
bottom = image.height * 0.75

cropped = image.crop((left, top, right, bottom))
cropped.save("5_cropped.jpg")

# 6. SHEARING
image = io.imread("image.jpg")
t = transform.AffineTransform(shear=0.5)  # change the sign to -0.5 for y shear

sheared = transform.warp(image, t)
io.imsave("6_sheared.jpg", (sheared * 255).astype("uint8"))


# ============================================================================
#                              EXERCISE 2
#                       Image Enhancement
# ============================================================================

from PIL import Image, ImageEnhance  # pip install Pillow

# Opens the image file
image = Image.open('image.jpg')

# 1. Enhance Brightness
curr_bri = ImageEnhance.Brightness(image)
img_brightened = curr_bri.enhance(2.5)
img_brightened.show()

# 2. Enhance Color Level
curr_col = ImageEnhance.Color(image)
img_colored = curr_col.enhance(2.5)
img_colored.show()

# 3. Enhance Contrast
curr_con = ImageEnhance.Contrast(image)
img_contrasted = curr_con.enhance(0.3)
img_contrasted.show()

# 4. Enhance Sharpness
curr_sharp = ImageEnhance.Sharpness(image)
img_sharped = curr_sharp.enhance(8.3)
img_sharped.show()


# ============================================================================
#                              EXERCISE 3
#                   Color Conversion and Segmentation
# ============================================================================

from skimage import data  # pip install scikit-image
from skimage.color import rgb2gray, rgb2hsv  # pip install scikit-image
import matplotlib.pyplot as plt  # pip install matplotlib

# Load the image
coffee = data.coffee()

# 1 RGB TO GRAYSCALE
gray_coffee = rgb2gray(coffee)
plt.imshow(gray_coffee, cmap='gray')
plt.show()

# 2 RGB TO HSV
hsv_coffee = rgb2hsv(coffee)
plt.imshow(hsv_coffee)
plt.show()

# 3 SUPERVISED SEGMENTATION USING THRESHOLDING
for i in range(10):
    threshold = i * 0.1
    binarized = gray_coffee > threshold

    plt.subplot(5, 2, i + 1)
    plt.title(f"Threshold: >{threshold: .1f}")
    plt.imshow(binarized, cmap='gray')

plt.tight_layout()
plt.show()


# ============================================================================
#                              EXERCISE 4
#                       Feature Extraction
# ============================================================================

# A. IMPORTING THE REQUIRED LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import filters, feature
from skimage.exposure import histogram

# B. LOADING THE IMAGE
# Load colored image
image1 = imread('image.jpg')
imshow(image1)
plt.show()

# Load grayscale image
image2 = imread('image.jpg', as_gray=True)
imshow(image2)
plt.show()

# C. ANALYZING BOTH THE IMAGES
# Shape of images
print("Colored image shape:", image1.shape)
print("Grayscale image shape:", image2.shape)

# Size of images
print("Colored image size:", image1.size)
print("Grayscale image size:", image2.size)

# D. FEATURE EXTRACTION

# I. PIXEL FEATURES
# Grayscale image pixel features
pixel_feat1 = np.reshape(image2, (3000 * 6000))  # Assuming image size is 3000x6000
print("Grayscale pixel features:", pixel_feat1)

# Colored image pixel features
pixel_feat2 = np.reshape(image1, (3000 * 6000 * 3))
print("Colored pixel features:", pixel_feat2)

# II. EDGE FEATURES
# Prewitt Kernel
pre_hor = filters.prewitt_h(image2)
pre_ver = filters.prewitt_v(image2)

# Sobel Kernel
ed_sobel = filters.sobel(image2)

# Canny Algorithm
can = feature.canny(image2)

# Display edge features
imshow(pre_ver, cmap='gray')
plt.title('Prewitt Vertical')
plt.show()

imshow(ed_sobel, cmap='gray')
plt.title('Sobel')
plt.show()

imshow(can, cmap='gray')
plt.title('Canny')
plt.show()

# III. REGION-BASED SEGMENTATION
hist, hist_centers = histogram(image2)

# Plotting the Image and the Histogram of gray values
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(image2, cmap='gray')
axes[1].plot(hist_centers, hist, lw=2)
axes[1].set_title('histogram of gray values')
plt.show()

# ============================================================================
