#date: 2022-01-12T17:17:53Z
#url: https://api.github.com/gists/372793fcea5f51ab383b347d8f0a6784
#owner: https://api.github.com/users/stasfilin

import cv2
import numpy as np

def check_image(image, new_height, new_width):
    height = image.shape[0]
    width = image.shape[1]
    colour = image.shape[2]

    BLACK_COLOUR = [0] * colour

    if height == new_height and width == new_width:
        return image

    imageList = image.tolist()

    one_column = [BLACK_COLOUR] * new_width

    for row in imageList:
        if len(row) < new_width:
            row.extend([BLACK_COLOUR] * (new_width - len(row)))
    imageList.extend([one_column] * (new_height - len(imageList)))

    return np.array(imageList)


blank_image = cv2.imread('blank_image.png')
new_image = check_image(blank_image, 120, 150)

cv2.imwrite('new_image.png', new_image)