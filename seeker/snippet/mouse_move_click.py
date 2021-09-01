#date: 2021-09-01T01:43:54Z
#url: https://api.github.com/gists/d9742b87964356dd8bd2922a32ad37b7
#owner: https://api.github.com/users/aont

#!/usr/bin/env python

import sys
import enum
import os
import time

# http://timgolden.me.uk/pywin32-docs/contents.html
import pywintypes
import win32api
import win32con


screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
sys.stderr.write("screen: %s %s\n" % (screen_width, screen_height))

mouse_pos_list = (
    (1, 1),
    (1, screen_height - 1),
    (screen_width - 1, screen_height - 1),
    (screen_width - 1, 1),
)

continue_flag = True


mouse_pos_startmenu = (1, screen_height - 1)
while True:

    for i in range(2):
        win32api.SetCursorPos((1,1))
        time.sleep(0.1)

        win32api.SetCursorPos(mouse_pos_startmenu)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    time.sleep(1)

    mouse_pos_current = win32api.GetCursorPos()
    if mouse_pos_current != mouse_pos_startmenu:
        break
    # win32api.keybd_event(win32con.VK_ESCAPE, 0, 0, 0)
    # win32api.keybd_event(win32con.VK_ESCAPE, 0, win32con.KEYEVENTF_KEYUP, 0)

