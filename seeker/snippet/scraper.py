#date: 2021-12-09T17:17:37Z
#url: https://api.github.com/gists/f2df88304f6bcc6354a461471af5c6cb
#owner: https://api.github.com/users/TomerAdmon

##
## This code will download frames from a live stream and save them as "frame%d.jpg" in your disk"
## make sure you are using python3 and downloading the opencv-python package
##

import cv2

vidcap = cv2.VideoCapture('https://5d8c50e7b358f.streamlock.net/live/EVLAIM.stream/chunklist_w117438879.m3u8')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1