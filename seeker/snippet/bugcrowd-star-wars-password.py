#date: 2022-05-06T17:16:05Z
#url: https://api.github.com/gists/cf8115791f37d6333307a6fd3a8f85b0
#owner: https://api.github.com/users/inkyvoxel

# https://twitter.com/Bugcrowd/status/1522268643719102466

import re

translate = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "J": 10,
    "K": 11,
    "L": 12,
    "M": 13,
    "N": 14,
    "O": 15,
    "P": 16,
    "Q": 17,
    "R": 18,
    "S": 19,
    "T": 20,
    "U": 21,
    "V": 22,
    "W": 23,
    "X": 24,
    "Y": 25,
    "Z": 26,
}

phrases = [
    "aaa thisaaa shoulda matchaaaa",
    "wont match"
]

for phrase in phrases:
    if re.match("^[^\s]{3}\s{1}[^\s]{7}\s{1}[^\s]{7}\s{1}[^\s]{9}$", phrase):
        encoded = ""
        for character in phrase:
            if character == " ":
                continue

            encoded += str(translate[str(character).upper()]) + " "

        print(f"{phrase}: {encoded}")