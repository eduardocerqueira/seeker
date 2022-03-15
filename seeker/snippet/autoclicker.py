#date: 2022-03-15T17:12:26Z
#url: https://api.github.com/gists/3f7df83ab8c8e503fa617049e85f43ea
#owner: https://api.github.com/users/bigjango13

from time import sleep
from pynput.mouse import Button, Controller
import keyboard

mouse = Controller()

# Change these to edit the keybindings
leftClickKeybindings = ['c', 'v'] 
rightClickKeybindings = ['x', 'c']
toggleClickKeybindings = ['z', 'x']
exitKeybindings = ['esc', '1']
delay = .005

while True:
    if keyboard.is_pressed(exitKeybindings[0]) and keyboard.is_pressed(exitKeybindings[1]):
        quit()
    else:
        while keyboard.is_pressed(leftClickKeybindings[0]) and keyboard.is_pressed(leftClickKeybindings[1]): # Spam left click
            mouse.press(Button.left)
            sleep(delay)
            mouse.release(Button.left)
            sleep(delay)
        while keyboard.is_pressed(rightClickKeybindings[0]) and keyboard.is_pressed(rightClickKeybindings[1]): # Spam right click
            mouse.press(Button.right)
            sleep(delay)
            mouse.release(Button.right)
            sleep(delay)
        while keyboard.is_pressed(toggleClickKeybindings[0]) and keyboard.is_pressed(toggleClickKeybindings[1]): # Spam right and then left click
            mouse.press(Button.right)
            sleep(delay)
            mouse.release(Button.right)
            sleep(delay)
            mouse.press(Button.left)
            sleep(delay)
            mouse.release(Button.left)
            sleep(delay)