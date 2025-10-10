#date: 2025-10-10T17:11:59Z
#url: https://api.github.com/gists/4699b92a367f1bf79bc099799254e8ca
#owner: https://api.github.com/users/TransparentLC

import base64
import hmac
import os
import time
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

# maimai.py/maimai-ffi 直接使用了 cryptography.Fernet 对用户 ID 进行加密
# 实际的密文格式如下
# https://github.com/fernet/spec/blob/master/Spec.md
# 密钥在 maimai-ffi 的二进制文件中，直接 strings 即可找到

MAIMAI_PY_KEY = base64.urlsafe_b64decode(R'C4H2HV3vIbMFzLJvosbie9qDEqH32zkqPZYr8cJUiDg=')
MAIMAI_PY_KEY_HMAC, MAIMAI_PY_KEY_AES = MAIMAI_PY_KEY[:16], MAIMAI_PY_KEY[16:]

def encode_arcade_credential(userId: int, timestamp: int | float | None = None) -> str:
    '''
    将用户 ID 转换为 maimai.py 使用的加密的机台 PlayerIdentifier 的 credentials

    Parameters
    ----------
    userId : int
        用户 ID
    timestamp : int | float | None, optional
        credentials 中的时间戳，留空或默认则使用当前时间

    Returns
    -------
    str
        maimai.py 使用的加密的机台 PlayerIdentifier 的 credentials

        PlayerIdentifier(credentials=...)
    '''
    if timestamp is None:
        timestamp = time.time()
    timestamp: bytes = int(timestamp).to_bytes(8, 'big')
    iv = os.urandom(16)
    plaintext = userId.to_bytes(8, 'big')
    ciphertext = AES.new(MAIMAI_PY_KEY_AES, AES.MODE_CBC, iv=iv).encrypt(pad(plaintext, 16))
    hmacData = b'\x80' + timestamp + iv + ciphertext
    hmacTag = hmac.digest(MAIMAI_PY_KEY_HMAC, hmacData, 'sha256')
    return base64.urlsafe_b64encode(hmacData + hmacTag).decode()

def decode_arcade_credential(credential: str) -> int:
    '''
    将 maimai.py 使用的加密的机台 PlayerIdentifier 的 credentials 转换为用户 ID

    Parameters
    ----------
    credential : str
        maimai.py 使用的加密的机台 PlayerIdentifier 的 credentials

        PlayerIdentifier(credentials=...)

    Returns
    -------
    int
        用户 ID
    '''
    credential: bytes = base64.urlsafe_b64decode(credential)
    hmacData, hmacTag = credential[:-32], credential[-32:]
    version, timestamp, iv, ciphertext = hmacData[0], hmacData[1:9], hmacData[9:25], hmacData[25:]
    assert version == 0x80 and hmac.compare_digest(hmacTag, hmac.digest(MAIMAI_PY_KEY_HMAC, hmacData, 'sha256'))
    return int.from_bytes(unpad(AES.new(MAIMAI_PY_KEY_AES, AES.MODE_CBC, iv=iv).decrypt(ciphertext), 16))
