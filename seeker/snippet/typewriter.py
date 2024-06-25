#date: 2024-06-25T17:05:16Z
#url: https://api.github.com/gists/920dea27d86546f32b6e3890069700d8
#owner: https://api.github.com/users/Pr1meSuspec7

import sys, time

def typewriter(text):
	for character in text:
		sys.stdout.write(character)
		sys.stdout.flush()
		time.sleep(0.008)
	print("\n")
