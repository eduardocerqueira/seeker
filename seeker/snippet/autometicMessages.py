#date: 2023-04-25T16:40:16Z
#url: https://api.github.com/gists/db4d7e482d59212d819a35f0653281cd
#owner: https://api.github.com/users/angurumouli

import pyautogui
import time

# Wait for 5 seconds before sending the message
time.sleep(5)

# Type the message and press Enter
count =0;
while count<=15:
    pyautogui.typewrite("Hello, this is an automatic message sent using Python!")
    pyautogui.press("enter")
    count=count+1;

# // sudo apt install python3-pip
# //pip install pyautogui
# //python3 scamm_2023.py or <File Name>