#date: 2023-07-24T16:52:25Z
#url: https://api.github.com/gists/e30547fb0f0f4cfa39a86b3a9b9698ad
#owner: https://api.github.com/users/DeathWingDuck

import os
import numpy
import cv2
import glob


# Load data

cropped_images = "Exported"

if not os.path.exists(cropped_images):
    os.makedirs(cropped_images)

# Load data
filepath = "C:\\Photos\\"
orig_files = [file for file in glob.glob(filepath+"/*.jpg")]
new_files = [os.path.join(cropped_images, os.path.basename(f)) for f in orig_files]

for orig_f, new_f in zip(orig_files, new_files):
    img = cv2.imread(orig_f)
    tile = numpy.tile(img, (1, 2, 1))


    cv2.imwrite(new_f, tile)
