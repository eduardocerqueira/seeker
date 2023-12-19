#date: 2023-12-19T16:38:50Z
#url: https://api.github.com/gists/25b324c077ed11d2e66cb2226c6880dc
#owner: https://api.github.com/users/robinhouston

# -*- encoding: utf-8 -*-

import random
import re

POSITIVE_DIRS = [ (0,1), (1,1), (1,0), (1,-1) ]
DIRS = POSITIVE_DIRS + [ (-dx, -dy) for (dx, dy) in POSITIVE_DIRS ]

class WordSearch(object):
	def __init__(self, width, height, grid=None):
		self.width = width
		self.height = height

		self.grid = [ [" " for x in range(width)] for y in range(height) ]
		if grid is not None:
			for y in range(height):
				for x in range(width):
					self.grid[y][x] = grid[y][x]

	@classmethod
	def load(cls, f):
		grid = [
			list(re.sub(r"[^a-z]", "", line.lower()))
			for line in f
			if re.match(r"[a-z]", line, flags=re.I)
		]

		return cls(len(grid[0]), len(grid), grid)

	def words_in_dir_through(self, word_len, dir, ox, oy):
		for i in range(word_len):
			x = ox - dir[0] * i
			y = oy - dir[1] * i

			ex, ey = x + dir[0]*(word_len - 1), y + dir[1]*(word_len - 1)
			if 0 <= x < self.width and 0 <= ex < self.width and 0 <= y < self.height and 0 <= ey < self.height:
				yield "".join([ self.grid[y + j*dir[1]][x + j*dir[0]] for j in range(word_len) ])

	def words_through(self, word_len, x, y):
		for dir in DIRS:
			for word in self.words_in_dir_through(word_len, dir, x, y):
				yield word

	def words_in_dir_at(self, word_len, dir, x, y):
		ex, ey = x + dir[0]*(word_len - 1), y + dir[1]*(word_len - 1)
		if 0 <= x < self.width and 0 <= ex < self.width and 0 <= y < self.height and 0 <= ey < self.height:
			yield x, y, "".join([ self.grid[y + j*dir[1]][x + j*dir[0]] for j in range(word_len) ])

	def words_at(self, word_len, x, y):
		for dir in DIRS:
			for sx, sy, word in self.words_in_dir_at(word_len, dir, x, y):
				yield sx, sy, dir, word

	def insert_at_random(self, word):
		dir = random.choice(DIRS)
		while True:
			x = random.randrange(
				len(word) if dir[0] == -1 else 0,
				self.width - len(word) if dir[0] == +1 else self.width)
			y = random.randrange(
				len(word) if dir[1] == -1 else 0,
				self.height - len(word) if dir[1] == +1 else self.height)

			try_again = False
			for letter in word:
				if self.grid[y][x] not in (" ", letter):
					try_again = True
					break
				self.grid[y][x] = letter
				x += dir[0]
				y += dir[1]

			if not try_again: break

	def count(self):
		n = 0
		for row in self.grid:
			for letter in row:
				if letter != " ":
					n += 1
		return n

	def __getitem__(self, y):
		return self.grid[y]

	def __iter__(self):
		return iter(self.grid)

	def __str__(self):
		return "\n".join([
			" ".join(row)
			for row in self.grid
		])
