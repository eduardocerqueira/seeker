#date: 2023-09-12T16:59:09Z
#url: https://api.github.com/gists/5bce4fb4691edce78e063efee908c6b9
#owner: https://api.github.com/users/ArrayOfLilly

# student_dict = {
#     "student": ["Angela", "James", "Lily"],
#     "score": [56, 76, 98]
# }
#
# # Looping through dictionaries:
# for (key, value) in student_dict.items():
#     # Access key and value
#     pass

import pandas

# student_data_frame = pandas.DataFrame(student_dict)
#
# # Loop through rows of a data frame
# for (index, row) in student_data_frame.iterrows():
#     # Access index and row
#     # Access row.student or row.score
#     pass

# Keyword Method with iterrows()
# {new_key:new_value for (index, row) in df.iterrows()}

# TODO 1. Create a dictionary in this format:
# {"A": "Alfa", "B": "Bravo"}

csv_input = pandas.read_csv("nato_phonetic_alphabet.csv")
# print(csv_input)
alphabet = {row.letter: row.code for (index, row) in csv_input.iterrows()}
# print(alphabet)

# TODO 2. Create a list of the phonetic code words from a word that the user inputs.

word_to_spell = input("Which word would you like to spell? ").upper()
# spell = {letter: code for (letter, code) in alphabet.items() if letter in word_to_spell}
spelled_word = {letter:alphabet[letter] for letter in word_to_spell}

print(spelled_word)
