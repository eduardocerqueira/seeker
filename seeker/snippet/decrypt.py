#date: 2022-11-04T17:03:32Z
#url: https://api.github.com/gists/1c53b38943a28c111067b73dbf542f32
#owner: https://api.github.com/users/code-and-dogs

def decryption(file, otpKey):
    encryptedFile = open(file, 'rb').read()
    otpKey = open(otpKey, 'rb').read()
    decryptedFile = bytes (a ^ b for (a, b) in zip(encryptedFile, otpKey))
    with open(file, 'wb') as decrypted:
        decrypted.write(decryptedFile)