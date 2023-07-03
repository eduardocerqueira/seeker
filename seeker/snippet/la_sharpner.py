#date: 2023-07-03T17:05:12Z
#url: https://api.github.com/gists/e4a9eaca8a7173a90d1c9cfae85e5a36
#owner: https://api.github.com/users/ajratnam

# Import the required modules
try:
    import cv2
    import numpy as np
except ImportError as err:
    print("Please install the modules python-opencv and numpy")
    raise err

# Load the image
image = cv2.imread('photo.jpeg', cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

# Convert it to YUV color space
image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

# Split the image into it's individual Y, U, V channels
y, u, v = cv2.split(image_yuv)
height, width = y.shape[:2]


def resize(*axes):
    return [*map(lambda axis: cv2.resize(axis, (width, height)).astype(np.float32), axes)]


# Resize the channel to match the image's dimensions
y, u, v = resize(y, u, v)

# Create the Kernel Matrix
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], dtype=np.float32)

# Normalize the Matrix
kernel /= np.sum(kernel)

# Pad the sides with zero's to handle the border pixels
padded_y_space = np.pad(y, pad_width=1, mode='constant')
final_y_space = np.zeros_like(y)

# Convolute each pixel with the Kernel Matrix
for i in range(1, height + 1):
    for j in range(1, width + 1):
        final_y_space[i - 1, j - 1] = np.sum(padded_y_space[i - 1:i + 2, j - 1:j + 2] * kernel)

# Clip all the pixels so that it stays within the 0 - 255 range
clipped_y_space = np.clip(final_y_space, 0, 255)

# Merge the new convoluted Y channel with the remaining U and V channels
filtered_image_yuv = cv2.merge((clipped_y_space, *resize(u, v))).astype(np.uint8)
# Convert the YUV image back to the RGB color space
filtered_image_bgr = cv2.cvtColor(filtered_image_yuv, cv2.COLOR_YUV2BGR)

# Display the images
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', filtered_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
