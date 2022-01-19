#date: 2022-01-19T17:06:17Z
#url: https://api.github.com/gists/7eb6f732e05d9d23850df0afaeadb9a5
#owner: https://api.github.com/users/nick3499

#!/bin/python3
'''Encode or decode ROT47'''
# Any spaces are left unchanged.


def rot47(string):
    '''Encode/decode string.'''
    enc_dec = ''  # string encoded/decoded to/from ROT47
    for char in string:
        ascii_code = ord(char)  # convert character to ASCII code number
        if ascii_code >= 33 and ascii_code <= 79:
            enc_dec += chr(ascii_code + 47)
        elif ascii_code >= 80 and ascii_code <= 126:
            enc_dec += chr(ascii_code - 47)
        else:
            enc_dec += char
    return enc_dec


if __name__ == '__main__':
    print(rot47(input('Enter string: ')))
