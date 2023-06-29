#date: 2023-06-29T16:57:33Z
#url: https://api.github.com/gists/89566afe5643cc77725637fb816bb9af
#owner: https://api.github.com/users/TheEntropyShard

"""
Decrypt HARMAN AIR SDK encrypted swfs
"""

from struct import pack, unpack
from Crypto.Cipher import AES
import sys

globalKey = b"Adobe AIR SDK (c) 2021 HARMAN Internation Industries Incorporated"

def getkey(data):
    dsum = sum(data) # sum all the bytes of data
    dmod = dsum % len(globalKey)
    
    string = globalKey[dmod:] + globalKey[:dmod]
    string += b" EncryptSWF "
    string += str(dsum).encode()
    
    ret = 0
    for code in string:
        ret *= 31
        ret += code
        
    return ret & 0xFFFFFFFF # it's an int
    
    
def decrypt(data):
    # read the different components
    data = bytearray(data)
    data[0] -= 32 # capitalize it
    
    # get the key
    key = getkey(data[0:8])
    
    # get the length
    decryptedLength = unpack("<I", data[8:12])[0]
    decryptedLength ^= key
    
    # padded length
    paddedLength = (decryptedLength + 0x1F) & ~0x1F
    
    # aes iv
    aesIV = bytearray(16)
    aesIV[0:8] = data[0:8] # header
    aesIV[8:12] = data[8:12] # encrypted length
    aesIV[12:16] = pack("<I", key) # key, uint32
    
    # xor in second half of iv
    for i in range(16):
        aesIV[i] ^= globalKey[i]
    
    # aes key
    # this one is stored at the end of the file
    aesKey = bytearray(32)
    aesKeyIdx = 8 + 4 + paddedLength # skip header, size, data
    
    # decrypt aes key
    for i in range(0, 32, 4):
        value = unpack("<I", data[aesKeyIdx + i : aesKeyIdx + i + 4])[0] # from uint32
        if i & 4:
            value -= key
        else:
            value += key
            
        aesKey[i:i+4] = pack("<I", value & 0xFFFFFFFF)
        
    # we've got our aes key
    aes = AES.new(aesKey, AES.MODE_CBC, IV=aesIV)
    print("aes key", aesKey.hex(" "))
    print("aes iv", aesIV.hex(" "))
    
    decrypted = aes.decrypt(data[12:12+paddedLength])
    return data[:8] + decrypted[:decryptedLength]
    
    
with open(sys.argv[1], "rb") as f:
    data = f.read()
    
with open(sys.argv[2], "wb") as f:
    f.write(decrypt(data))