#date: 2023-01-09T17:08:11Z
#url: https://api.github.com/gists/bb7be6d813931f4b15f32876eb5f55e3
#owner: https://api.github.com/users/secemp9

import win32api
from PIL import ImageGrab

# import matplotlib.pyplot as plt # just for testing, isn't needed
# import matplotlib.image as mpimg

def grab(monitor_number):
    _, _, bbox_old = win32api.EnumDisplayMonitors()[monitor_number]
    print("this is the wrong one:", bbox_old)
    a = win32api.EnumDisplaySettings()
    bbox = (0, 0, a.PelsWidth, a.PelsHeight)
    print("this is the right one:", bbox)
    return ImageGrab.grab(bbox, all_screens=True)


# grab(0).save(f"1.png") # this works on my end, but for multi monitor testing...:
for i in range(3):
    grab(i).save(f"out/{i}.png")
# img = mpimg.imread("1.png") # just for testing, don't mind me
# imgplot = plt.imshow(img)
# plt.show()