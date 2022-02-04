#date: 2022-02-04T17:00:08Z
#url: https://api.github.com/gists/0a5319c19eff0156745f8cc6892bb38f
#owner: https://api.github.com/users/ponbac

from PIL import ImageGrab
import pyautogui
from time import sleep
from random import randint, random


def check_pixel(x, y):
    image = ImageGrab.grab()
    pixel = image.getpixel((x, y))
    print(pixel)

    return pixel

# Edit X, Y to pixel that will be changed upon entering the queue
X, Y = 1158, 457
COLOR = check_pixel(X, Y)

while True:
    pyautogui.moveTo(200, 250)
    pyautogui.click()
    pyautogui.press('enter')
    if check_pixel(X, Y) != COLOR:
        break
    sleep(randint(5, 10) + random())
    pyautogui.press('enter')
    sleep(randint(0, 2) + random())