#date: 2021-11-03T17:14:42Z
#url: https://api.github.com/gists/1e2f3701d774a4e5d847f309da67fbea
#owner: https://api.github.com/users/MattieOF

import os
import sys
from spellchecker import SpellChecker


def get_int_input(prompt, error="Input must be an integer!"):
    valid = False
    while not valid:
        try:
            x = int(input(prompt))
            valid = True
            return x
        except ValueError:
            print(error)


print("Word List Filter")

# Get base list file name
fileName = input("Filename of base word list (include ext): ")
fileExists = os.path.exists(fileName)
if not fileExists:
    print(f"File \"{fileName}\" does not exist!")
    sys.exit(1)

# Open file into array
baseFile = open(fileName, 'r')
allWords = baseFile.read().splitlines()  # Use splitlines instead of readlines to avoid \n

# Get word length
hasMinLength = input("Require minimum length (y/n): ")
if hasMinLength.lower() == "y":
    minimumLength = get_int_input("Minimum length of word: ")
    if minimumLength < 1:
        print("Value must be at least 1, defaulting to 1.")
        minimumLength = 1

# Declare filtered list
filteredWords = []

# Get only words of length
if hasMinLength == "y":
    for word in allWords:
        if len(word) == minimumLength:
            filteredWords.append(word)
else:
    filteredWords = allWords

# Get dictionary requirement
checkDictionary = input("Filter by spellchecker (y/n): ")
if checkDictionary.lower() == "y":
    spell = SpellChecker()
    prevLength = len(filteredWords)
    filteredWords = spell.known(filteredWords)
    print(f"Filtered out {prevLength - len(filteredWords)} words")

# Sort alphabetically
sort = input("Sort alphabetically (y/n): ")
if sort.lower() == "y":
    filteredWords.sort()

# Export filtered list to file
exportFilename = input("Filtered list filename (include ext): ")
exportFile = open(exportFilename, 'w')
for word in filteredWords:
    exportFile.write(f"{word}\n")

print(f"Wrote {len(filteredWords)} words to {exportFilename}")
