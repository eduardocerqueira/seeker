#date: 2022-12-14T16:58:51Z
#url: https://api.github.com/gists/55c91f1be163bc2bea03ec7617c2a54b
#owner: https://api.github.com/users/thirdratecyberpunk


import nltk
from nltk.corpus import words

def analysis_problem():
    """
    Finds all the even length words in the sentence
    """
    message = "The Director has written his usual Christmas message but now uncover the odd one out amongst the words forming these two sentences Its odd because its not odd"
    tokens = "**********"
    even = []
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"l "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********"  "**********"% "**********"  "**********"2 "**********"  "**********"= "**********"= "**********"  "**********"0 "**********": "**********"
            even.append(token)
    return even

def coding_problem():
    """
    Brute forces all the possible solutions to the grid problem
    Checks if all three rows are valid 3 letter word rows and then if the final state creates a 9 letter word
    """
    # get all words
    nltk.download('words') 
    word_list = words.words()

    # initial problem constraints
    initial_state = [["f", "o", "r"], ["m", "a", "t"], ["i", "o", "n"]]
    blue_word = "part"
    green_word = "eyes"
    gold_word = "uncurl"

    # finds all the valid words for row 0
    print("checking row 0...")
    row_0_valid_words = []
    for gold_character in gold_word:
        for blue_character in blue_word:
            row_0 = initial_state[0]
            row_0[0] = gold_character
            row_0[1] = blue_character
            result = ''.join(row_0)
            if result in words.words():
                row_0_valid_words.append(result)
    unique_row_0 = set(row_0_valid_words)
    print(f"{len(unique_row_0)} unique solutions found: {unique_row_0}")

    # finds all the valid words for row 1
    print("checking row 1...")
    row_1_valid_words = []
    for gold_character in gold_word:
        for blue_character in blue_word:
            for green_character in green_word:
                row_1 = initial_state[1]
                row_1[0] = blue_character
                row_1[1] = green_character
                row_1[2] = gold_character
                result = ''.join(row_1)
                if result in words.words():
                    row_1_valid_words.append(result)
    unique_row_1 = set(row_1_valid_words)
    print(f"{len(unique_row_1)} unique solutions found: {unique_row_1}")

    # finds all the valid words for row 2
    print("checking row 2...")
    row_2_valid_words = []
    for gold_character in gold_word:
        for blue_character in blue_word:
            for green_character in green_word:
                row_2 = initial_state[2]
                row_2[0] = blue_character
                row_2[1] = gold_character
                row_2[2] = green_character
                result = ''.join(row_2)
                if result in words.words():
                    row_2_valid_words.append(result)
    unique_row_2 = set(row_2_valid_words)
    print(f"{len(unique_row_2)} solutions found: {unique_row_2}")

    # finds a combination of valid words from the rows that creates a valid 9 letter word
    print("finding valid 9 letter word(s)...")
    solutions = []
    for row_0_valid in unique_row_0:
        for row_1_valid in unique_row_1:
            for row_2_valid in unique_row_2:
                candidate = row_0_valid + row_1_valid + row_2_valid
                if candidate in words.words():
                    solutions.append(candidate)
    return solutions

print(analysis_problem())
print(coding_problem())