#date: 2024-07-12T16:56:38Z
#url: https://api.github.com/gists/96229aa4f23117c03b4f05e6abfdeb65
#owner: https://api.github.com/users/KunYi

#!/usr/bin/python3

from Crypto.Cipher import AES
import binascii

def get_uid(uid_string):
    """
    Convert the UID string to lowercase and then to a byte array of specified length.

    Parameters:
    uid_string (str): The UID string to be converted.

    Returns:
    bytes: The converted byte array.
    """
    # Convert the string to lowercase
    lowercase_uid = uid_string.lower()

    # Take the required number of characters
    shortened_uid = lowercase_uid[:8]
    print(shortened_uid)

    # Convert the shortened string to a byte array
    byte_array_uid = shortened_uid.encode('ascii')
    return byte_array_uid


# my device uid '10063F1FB910E0DC'
#uid = b'10063f1f'
#uid = b'10093f30'
uid = get_uid('10063F1FB910E0DC')

# AES key "59494F4754fff00" store in firmware
key = b'59494F4754fff00\0'

aes = AES.new(key, AES.MODE_ECB)

dat = aes.encrypt(uid + uid)

otp = dat[:8].hex().encode('ascii').hex()
print(otp)

binotp = binascii.unhexlify(otp)
print(binotp)

buff = bytearray([0xFF] * 1024)
buff[0:16] = binotp
buff[256:256+15] = binotp[1:]
buff[512:512+14] = binotp[2:]
buff[768:768+13] = binotp[3:]

with open('otp.bin', 'wb') as f:
    f.write(buff)
print("Binary data written to otp.bin")