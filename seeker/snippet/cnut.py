#date: 2023-12-19T16:38:50Z
#url: https://api.github.com/gists/25b324c077ed11d2e66cb2226c6880dc
#owner: https://api.github.com/users/robinhouston

#!/usr/bin/env python3

TARGET = "cnut"
AVOID = { "cnut", "\x63\x75\x6E\x74" }
WIDTH = 20
HEIGHT = 10

DIRS = [ (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1, 1) ]

import random

grid = [ [" " for x in range(WIDTH)] for y in range(HEIGHT) ]

def words_in_dir_through(word_len, dir, ox, oy):
	for i in range(word_len):
		x = ox - dir[0] * i
		y = oy - dir[1] * i

		ex, ey = x + dir[0]*(word_len - 1), y + dir[1]*(word_len - 1)
		if 0 <= x < WIDTH and 0 <= ex < WIDTH and 0 <= y < HEIGHT and 0 <= ey < HEIGHT:
			yield "".join([ grid[y + j*dir[1]][x + j*dir[0]] for j in range(word_len) ])

def words_through(word_len, x, y):
	for dir in DIRS:
		for word in words_in_dir_through(word_len, dir, x, y):
			yield word

def has_forbidden_word(x, y):
	for word in words_through(4, x, y):
		if word in AVOID:
			return True
	return False

# Put in the target word at random
dir = random.choice(DIRS)
x = random.randrange(len(TARGET) if dir[0] == -1 else 0, WIDTH - len(TARGET) if dir[0] == +1 else WIDTH)
y = random.randrange(len(TARGET) if dir[1] == -1 else 0, HEIGHT - len(TARGET) if dir[1] == +1 else HEIGHT)
for letter in TARGET:
	grid[y][x] = letter
	x += dir[0]
	y += dir[1]

num_assigned = len(TARGET)
while num_assigned < WIDTH * HEIGHT:
	x, y = random.randrange(WIDTH), random.randrange(HEIGHT)
	if grid[y][x] == " ":
		letter = random.choice(TARGET)
		grid[y][x] = letter
		if has_forbidden_word(x, y):
			grid[y][x] = " "
		else:
			num_assigned += 1

for row in grid:
	print(" ".join(row))
