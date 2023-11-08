#date: 2023-11-08T16:54:16Z
#url: https://api.github.com/gists/d53253c46db9c40faf5085ef9dd299a9
#owner: https://api.github.com/users/bneises

import string
import random

def generate_password(length: "**********": int = 2, upper_min: int = 2, numeric_min: int = 2, special_min: int = 2, special_exclude: str = '') -> str:
    LOWER = string.ascii_lowercase
    UPPER = string.ascii_uppercase
    NUMERIC = string.digits
    SPECIAL = string.punctuation

    for exclusion in special_exclude.split():
        SPECIAL = SPECIAL.replace(exclusion, '')

    min_length = sum([lower_min, upper_min, numeric_min, special_min])
    if length < min_length:
        print('Extending length to include all min character limits.')
        length = min_length

    list_of_characters = []
    list_of_characters.extend(random.choices(LOWER, k=lower_min))
    list_of_characters.extend(random.choices(UPPER, k=upper_min))
    list_of_characters.extend(random.choices(NUMERIC, k=numeric_min))
    list_of_characters.extend(random.choices(SPECIAL, k=special_min))
    list_of_characters.extend(random.choices(
        LOWER + UPPER + NUMERIC + SPECIAL, 
        k=length - len(list_of_characters)
    ))
    random.shuffle(list_of_characters)
    return ''.join(list_of_characters)

if __name__ == '__main__':
    print(generate_password())word())