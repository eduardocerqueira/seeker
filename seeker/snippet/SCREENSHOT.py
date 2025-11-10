#date: 2025-11-10T17:13:31Z
#url: https://api.github.com/gists/12d5bec2c889eef3300f276882ab5874
#owner: https://api.github.com/users/gvbmafia

import pyautogui
import pyinput
import os
import pwinput
import time
camera = pyautogui.screenshot()
camera.save(f'fisier_1.png')
constant = "PAROLA"
parola = pwinput.pwinput(prompt='Introdu parola: ', mask='*')
if parola == constant:
    print("Parola corecta")
    time.sleep(5)
    camera = pyautogui.screenshot()
    camera.save('fisier_2.png')
    time.sleep(5)
    camera = pyautogui.screenshot()
    camera.save('fisier_3.png')
    time.sleep(5)
    camera = pyautogui.screenshot()
    camera.save('fisier_4.png')
else:
    print("Parola gresita")