#date: 2024-05-27T16:54:06Z
#url: https://api.github.com/gists/bcc093b78526ec20d1fc8ca2246f79e4
#owner: https://api.github.com/users/Py0me

multiplication_table = {
		"zero": {"eight": 0, "five": 0, "four": 0, "nine": 0, "one": 0, "seven": 0,  "six": 0, "ten": 0, "three": 0, "two": 0, "zero": 0},
		"two": {"eight": 16, "five": 10, "four": 8, "nine": 18, "one": 2, "seven": 14, "six": 12, "ten": 20, "three": 6, "two": 4, "zero": 0},
		"three": {"eight": 24, "five": 15, "four": 12, "nine": 27, "one": 3, "seven": 21, "six": 18, "ten": 30, "three": 9, "two": 6, "zero": 0},
		"ten": {"eight": 80, "five": 50, "four": 40, "nine": 90, "one": 10, "seven": 70, "six": 60, "ten": 100, "three": 30, "two": 20, "zero": 0},
		"six": {"eight": 48, "five": 30, "four": 24, "nine": 54, "one": 6, "seven": 42, "six": 36, "ten": 60, "three": 18, "two": 12, "zero": 0},
		"seven": {"eight": 56, "five": 35, "four": 28, "nine": 63, "one": 7, "seven": 49, "six": 42, "ten": 70, "three": 21, "two": 14, "zero": 0},
		"one": {"eight": 8,  "five": 5, "four": 4, "nine": 9, "one": 1, "seven": 7, "six": 6, "ten": 10, "three": 3, "two": 2, "zero": 0},
		"nine": {"eight": 72, "five": 45, "four": 36, "nine": 81, "one": 9, "seven": 63, "six": 54, "ten": 90, "three": 27, "two": 18, "zero": 0},
		"four" : {"eight": 32, "five": 20, "four": 16, "nine": 36, "one": 4, "seven": 28, "six": 24, "ten": 40, "three": 12, "two": 8, "zero": 0},
		"five": {"eight": 40, "five": 25, "four": 20, "nine": 45, "one": 5, "seven": 35, "six": 30, "ten": 50, "three": 15, "two": 10, "zero": 0},
		"eight": {"eight": 64, "five": 40, "four": 32, "nine": 72, "one": 8, "seven": 56, "six": 48, "ten": 80, "three": 24, "two": 16, "zero": 0},
		}

def multiply(val1 : str, val2 : str):
	try: 
		res = multiplication_table[val1][val2])
		return res
	except:
		return "zero"