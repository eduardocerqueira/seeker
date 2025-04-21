#date: 2025-04-21T17:04:52Z
#url: https://api.github.com/gists/02444660970e28a8c72574c07ae0dc66
#owner: https://api.github.com/users/dsbdev9

import random, time, keyboard
from termcolor import colored
import pyautogui as key
import os
import threading
import signal
import sys

os.system('cls' if os.name == 'nt' else 'clear')  # clear screen on any OS

WAIT_BETWEEN_BLOCKS = 2 # In minutes

prefixes = ['calc', 'process', 'compute', 'eval', 'gen', 'transform', 'create']
suffixes = ['Data', 'Result', 'Value', 'Output', 'Sum', 'Product', 'Equation']

WEIGHT_BETWEEN_BLOCKS = 4

math_expressions = [
    'a + b',
    'x * y',
    'z / 2',
    'math.sqrt(a)',
    'math.sin(b)',
    'math.cos(a + b)',
    'math.tan(x * y)',
]

def generate_lua_code():
    function_name = random.choice(prefixes) + random.choice(suffixes)
    math_line = random.choice(math_expressions)
    
    # Generate the Lua code
    lua_code = f"""
function {function_name}(a,b,c,x,y,z)
local result = {math_line}
return(result)"""
    return lua_code

print(colored("[?]", "yellow"), "Press 5 to start generating code")

def type_random_block():
    gibberish_code = generate_lua_code()

    for char in gibberish_code:
        key.write(char)
        time.sleep(0.01)
        #if char == '\n':
        #    time.sleep(0.1)
            #pyautogui.press('enter')

    # down arrow twice to get to the end of the code
    key.press('down')
    time.sleep(0.1)
    key.press('down')
    time.sleep(0.1)

    key.press('enter')
    key.press('enter')

generating = False
t = None
def type_thread():
    while True:
        type_random_block()
        for i in range(WAIT_BETWEEN_BLOCKS*60):
            if not generating:
                print(colored("[!]", "red"), "Stopped generating code")
                return
            time.sleep(1)


while True:
    try:
        if keyboard.is_pressed('5'):
            generating = not generating 
            time.sleep(0.5)
            if generating:
                print(colored("[!]", "green"), "Generating code")
                time.sleep(2.5)
                t = threading.Thread(target=type_thread)
                t.start()
                print(colored("[?]", "yellow"), "Press 5 to stop generating code")
            else:
                # stop the thread
                if t and t.is_alive():
                    t.join()
    except KeyboardInterrupt:
        print(colored("[!]", "red"), "Received KeyboardInterrupt, stopping code generation")
        generating = False
        if t and t.is_alive():
            t.join()
        sys.exit(0)
