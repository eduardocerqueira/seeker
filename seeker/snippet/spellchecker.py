#date: 2022-07-05T17:14:23Z
#url: https://api.github.com/gists/f2f5ddb2ce0f64164719fae3d1aef4a7
#owner: https://api.github.com/users/shahzaibkhan

from spellchecker import SpellChecker

spell = SpellChecker()

word = input("Enter your Word : ")

if word in spell:
    print("Correct Spelling")
else:
    print("Incorrect Spelling")
