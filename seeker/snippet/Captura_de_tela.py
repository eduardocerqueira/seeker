#date: 2023-11-03T16:47:14Z
#url: https://api.github.com/gists/b5716789d8f14a52054acc0cf4ff12a2
#owner: https://api.github.com/users/lostquix

import cv2
import keyboard
import pyautogui
import numpy as np

fps = 30
tamanho_tela = tuple(pyautogui.size())

codec = cv2.VideoWriter_fourcc(*"XVID")
video = cv2.VideoWriter("video.avi", fps, tamanho_tela)

while True:
    frame = pyautogui.screenshot()
    frame = np.array(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(frame)

    if keyboard.is_pressed("esc"):
        break

video.release()
cv2.destroyAllWindows()
