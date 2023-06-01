#date: 2023-06-01T16:57:25Z
#url: https://api.github.com/gists/8479be139e189d63f2f5237b3e0e131c
#owner: https://api.github.com/users/MysticDemonX

import re


def split_string(text):
    string = input(text)
    return set(re.sub(' ', '', string).split(','))


string1 = split_string('Введите слова через запятую: ')
string2 = split_string(f'В списке {len(string1)} уникальных слов, теперь введите второй список через запятую с '
                       f'таким же количеством слов: ')
dictionary = dict(zip(string1, string2))
print(dictionary)
