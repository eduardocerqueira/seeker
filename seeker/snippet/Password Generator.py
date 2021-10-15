#date: 2021-10-15T17:02:43Z
#url: https://api.github.com/gists/ae83be4e643cffbb6133d559423c0f8e
#owner: https://api.github.com/users/KryllyxOfficial

import random
import time

letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
          'n','o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

cap_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N','O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

password = []

option = ['1', '2', '3']

while True:
    query = input("[F]ive, [T]en, or [S]ixteen character password?\n> ")
    if query == 'F':
        break
    
    elif query == 'T':
        break
    
    elif query == 'S':
        break
    
    else:
        print("\nUnrecognizedAnswer Error - Please input \"F\", \"T\", or \"S\".")
        time.sleep(2)
        continue

if query == 'F':
    for key in range(5):
        value = random.choice(option)
        if value == '1':
            password.append(f"{random.choice(letter)}")
            
        elif value == '2':
            password.append(f"{random.choice(cap_letter)}")
            
        elif value == '3':
            password.append(f"{random.choice(number)}")
    
    generated = ''.join([str(strng) for strng in password])
        
    print(f"\nHere is your generated password:\n{generated}")

elif query == 'T':
    for key in range(10):
        value = random.choice(option)
        if value == '1':
            password.append(f"{random.choice(letter)}")
            
        elif value == '2':
            password.append(f"{random.choice(cap_letter)}")
            
        elif value == '3':
            password.append(f"{random.choice(number)}")
    
    generated = ''.join([str(strng) for strng in password])
        
    print(f"\nHere is your generated password:\n{generated}")

elif query == 'S':
    for key in range(16):
        value = random.choice(option)
        if value == '1':
            password.append(f"{random.choice(letter)}")
            
        elif value == '2':
            password.append(f"{random.choice(cap_letter)}")
            
        elif value == '3':
            password.append(f"{random.choice(number)}")
    
    generated = ''.join([str(strng) for strng in password])
        
    print(f"\nHere is your generated password:\n{generated}")