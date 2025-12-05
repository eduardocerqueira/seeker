#date: 2025-12-05T16:57:05Z
#url: https://api.github.com/gists/12cb2bd3237b41ef4cd898a62df0989c
#owner: https://api.github.com/users/d1toon

import string

start, end = input().split('-')
letters = string.ascii_letters
print(letters[letters.index(start):letters.index(end)+1])
