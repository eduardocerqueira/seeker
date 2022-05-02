#date: 2022-05-02T17:02:35Z
#url: https://api.github.com/gists/b3e87bf1fdb986ba82eea00106f3a507
#owner: https://api.github.com/users/m33ch33

#!/usr/bin/env python
# -*- coding: utf-8 -*-  
#
# This Hackvertor script shows basic usage of symmetric encryption.
# We would use it when certain parameters should be sent encrypted. 
# In this particular example using AES/CBC/PKCS5PADDING suit with a 
# padded timestamp as an initialization vector.

import base64
import hashlib
from java.util import Base64
from javax.crypto import Cipher
from javax.crypto.spec import IvParameterSpec
from javax.crypto.spec import SecretKeySpec

timestamp = "000000"+ str(ts)
# timestamp = "0000001631634319"

# either static or external value
encryptionKey = "theKey"
inCVV = str(input)

_cipherTransformation = 'AES/CBC/PKCS5PADDING'
_aesEncryptionAlgorithem = 'AES'

encryptedText = ''
cipher = Cipher.getInstance(_cipherTransformation)
key = encryptionKey.encode('utf-8')
secretKey = SecretKeySpec(key,'AES')
ivParameterSpec = IvParameterSpec(timestamp)
cipher.init(Cipher.ENCRYPT_MODE,secretKey,ivParameterSpec)
cipherText = cipher.doFinal(inCVV.encode('UTF-8'))
encoder = Base64.getEncoder()
encryptedText = encoder.encodeToString(cipherText)
output = encryptedText

# For local debugging
# print(encryptedText)
# return encryptedText