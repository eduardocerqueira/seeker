#date: 2025-06-18T17:16:21Z
#url: https://api.github.com/gists/842ba287114e4663a1122491e4d39a97
#owner: https://api.github.com/users/Keylightgithub

'''
prerequisites: 
    1. Have Windows 11 Snipping Tool open and an image snipped.
    2. Have Google Keep & Docs already signed in in your (default) browser.
'''

import pyautogui  # GUI automation: keyboard/mouse control.
import pygetwindow as gw  # Window management: find/activate windows.
import time  # For script delays and timing.
import sys  # System-specific functions (e.g., sys.exit).
import webbrowser  # Opens URLs in the default browser.


# Find the Snipping Tool window
time.sleep(1)
windows = gw.getWindowsWithTitle('Snipping Tool')
if windows:
    win = windows[0]
    print("Snipping Tool window found. Focusing...")
    win.activate()
    time.sleep(0.5)
    pyautogui.hotkey('ctrl', 'c')

    webbrowser.open("https://keep.google.com")
    time.sleep(3)
    pyautogui.hotkey('ctrl', 'f')
    time.sleep(0.2)
    pyautogui.typewrite("Take a note...")
    time.sleep(0.2)
    pyautogui.hotkey('enter')
    time.sleep(0.2)
    pyautogui.hotkey('escape')
    time.sleep(0.2)
    pyautogui.hotkey('enter')
    time.sleep(0.2)   
    pyautogui.hotkey('ctrl', 'v')
    time.sleep(0.2)
    print("Image pasted into Google Keep.")
    
    #select (text extract) button and (open in doc) button
    for _ in range(7):
        pyautogui.press('tab')
        time.sleep(0.2) # Small delay between tab presses
    time.sleep(5) # Wait for image to be uploaded to Google Keep
    pyautogui.hotkey('enter')
    
    for _ in range(3):
        pyautogui.press('up')
        time.sleep(0.2)
    time.sleep(0.2)
    pyautogui.hotkey('enter')
    time.sleep(0.2)
    pyautogui.hotkey('enter')
    time.sleep(0.2)
    
    for _ in range(2):
        pyautogui.press('up')
        time.sleep(0.2)
    time.sleep(0.2)
    pyautogui.hotkey('enter')
    time.sleep(5) # wait for doc output to be generated
    
    # open the doc output
    pyautogui.hotkey('ctrl', 'f')
    time.sleep(0.2)
    pyautogui.typewrite("Open Doc")
    time.sleep(0.2)
    pyautogui.hotkey('enter')
    time.sleep(0.2)
    pyautogui.hotkey('escape')
    time.sleep(0.2)
    pyautogui.hotkey('enter')
    time.sleep(0.2)
    # Wait for the document to open
    time.sleep(5)

    # attempt to select text portion in the document
    for _ in range(2):
        pyautogui.press('down')
        time.sleep(0.2)
    time.sleep(0.2)
    #pyautogui.hotkey('ctrl', 'shift', 'end') # doesn't work


else:
    print("Snipping Tool window not opened or no image snipped. Exiting script.")
    sys.exit() # Exit the script if the window is not found

print("Script completed.")
