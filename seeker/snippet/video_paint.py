#date: 2024-01-30T16:52:52Z
#url: https://api.github.com/gists/687ba5800a2757786d5d2c8f853e0670
#owner: https://api.github.com/users/NickyAlan

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import pyautogui as pg
from time import sleep
import win32clipboard as w32c 

VIDEO_PATH = "mn.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
frames_list = []
# Read and process each frame
i = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if i%2 == 0 : 
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_list.append(np.array(rgb_frame))
    i+=1

def copy2clipboard(content) :
    # https://stackoverflow.com/questions/34322132/copy-image-to-clipboard
    w32c.OpenClipboard()
    w32c.EmptyClipboard()
    w32c.SetClipboardData(w32c.CF_DIB, content)
    w32c.CloseClipboard()

paste_pos = (41, 96) #  position that paste btn at
pg.click(paste_pos)
for idx, array in enumerate(frames_list) :
    image = Image.fromarray(array)
    output = BytesIO()
    image.convert("RGB").save(output,  "BMP")
    content = output.getvalue()[14: ]
    output.close()
    copy2clipboard(content)
    pg.click(paste_pos)
    # move to center : 1302x738 pixel(paint) with 1280x720 video
    for _ in range(8) :
        pg.press("right")
        pg.press("down")
    sleep(1)
