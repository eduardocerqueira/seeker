#date: 2021-10-15T17:01:19Z
#url: https://api.github.com/gists/4bf382268c9a37f13c485c6a29fd744c
#owner: https://api.github.com/users/KryllyxOfficial

import random
import time

letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
          'n','o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def randomKey():
	key = []
	while True:
		key_length = input("\n\nEnter length of plaintext (excluding spaces and punctuation).\n> ")
		if key_length.isnumeric() == True:
			break
		
		else:
			print("\nUnrecogmizedAnswer Error - Please input a number")
			time.sleep(2)
			continue
	
	if key_length.isnumeric() == True:
		for char in range(int(key_length)):
			key.append(random.choice(letter))
		print(f"\nHere is your generated key:\n{''.join([str(strng) for strng in key])}")
        
randomKey()