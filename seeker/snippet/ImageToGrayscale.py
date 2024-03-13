#date: 2024-03-13T17:08:07Z
#url: https://api.github.com/gists/830a089e24a36b0cf6b4487813e47cc0
#owner: https://api.github.com/users/shemtomke

import cv2
import sys

# reading image and saving in a var
image = cv2.imread(sys.argv[1]) # The first argument is the image, getting from command line

print("image shape")
print(image.shape)

# Convert to Grayscale
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

'''First, we convert the image to gray. The function that does that is cvtColor().
 The first argument is the image to be converted, the second is the color mode.
 COLOR_BGR2GRAY stands  for Blue Green Red to Gray.
 You must have heard of the RGB color scheme. OpenCv does it the other way round-
 so blue is first, then green, then red.'''

cv2.imshow("Original Image", image)
cv2.imshow("Gray Image", grayImage)

cv2.waitKey(0)

'''

Note:

Two important functions in image processing are blurring and grayscale.

Many image processing operations take place on grayscale (or black and white) images, 
as they are simpler to process (having just two colors).

Similarly, blurring is also useful in edge detection, as we will see in next tut example. .

'''
