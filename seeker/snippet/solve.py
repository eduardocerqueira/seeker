#date: 2023-12-19T16:38:50Z
#url: https://api.github.com/gists/25b324c077ed11d2e66cb2226c6880dc
#owner: https://api.github.com/users/robinhouston

#!/usr/bin/env python3

import re
import sys

import wordsearch

def find(grid, word):
	reversed_word = "".join(reversed(word))
	for y in range(grid.height):
		for x in range(grid.width):
			for sx, xy, dir, found_word in grid.words_at(len(word), x, y):
				if found_word == word or found_word == reversed_word:
					for i in range(len(word)):
						grid[y + i*dir[1]][x + i*dir[0]] = grid[y + i*dir[1]][x + i*dir[0]].upper()


grid = wordsearch.WordSearch.load(sys.stdin)

for word in sys.argv[1:]:
	find(grid, word)

for row in grid:
	print(" ".join(row))
