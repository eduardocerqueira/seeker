#date: 2021-11-15T16:54:56Z
#url: https://api.github.com/gists/3e823c4beeaf82e2ae1898689c685cc6
#owner: https://api.github.com/users/dcbriccetti

import pathlib

# “really old way

wf2 = open('words.txt')
wf2.readlines()
wf2.close()

# Old” way, closes file automatically
with open('words.txt') as words_file:
    lines = words_file.readlines()

words: list[str] = [line.rstrip() for line in lines]  # List comprehension

# A better way

all_text: str = pathlib.Path('words.txt').read_text()
words = all_text.split('\n')
print(words)
