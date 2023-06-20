#date: 2023-06-20T16:44:31Z
#url: https://api.github.com/gists/cf587a0d837f92523137a7c69777eaca
#owner: https://api.github.com/users/Yunus-Guser

import pyautogui
import time

time.sleep(10)

f= open("spamlar.txt","r",encoding="utf-8")

for word in f:
    pyautogui.typewrite(word)
    pyautogui.press("enter")
    