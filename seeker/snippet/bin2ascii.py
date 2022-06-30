#date: 2022-06-30T16:53:18Z
#url: https://api.github.com/gists/249cfdcc1eb5e4daff2b121de00bdff6
#owner: https://api.github.com/users/jac18281828

import binascii
import sys

if  __name__ == '__main__':
    if len(sys.argv) > 1:
        file = sys.argv[1]
        with open(file, 'rb') as keyfile:
            private_key = keyfile.read()
            print(binascii.b2a_hex(private_key).decode('utf-8'))
    else:
        print('bin2ascii file')
