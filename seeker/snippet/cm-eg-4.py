#date: 2021-11-23T17:06:33Z
#url: https://api.github.com/gists/10983208d90a1da622ed5049f8c1c307
#owner: https://api.github.com/users/dichharai

import pathlib
from contextlib import contextmanager

path = pathlib.Path(__file__).parent.resolve()

@contextmanager
def open_file(file_name):
    print(f'opening {file_name}...')
    f = open(f'{path}/{file_name}')
    try:
        yield f
    finally:
        print(f'closing {file_name}...')
        f.close()

file_name = 'alice_1.txt'

with open_file(file_name) as f:
    while True:
        try:
            print(next(f))
        except StopIteration:
            break