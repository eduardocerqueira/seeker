#date: 2023-05-30T16:49:45Z
#url: https://api.github.com/gists/ea5d5526edc9861d80bab6a5723c20aa
#owner: https://api.github.com/users/Cozy03

KEY1 = 'a6c8b6733c9b22de7bc0253266a3867df55acde8635e19c73313'
KEY21 = '37dcb292030faa90d07eec17e3b1c6d8daf94c35d4c9191a5e1e'
KEY23 = 'c1545756687e7573db23aa1c3452a098b71a7fbf0fddddde5fc1'
RES = '04ee9855208a2cd59091d04767ae47963170d1660df7f56f5faf'

def xor_hex_strings(hex_str1, hex_str2):
    int_val1 = int(hex_str1, 16)
    int_val2 = int(hex_str2, 16)
    xor_result = int_val1 ^ int_val2
    xor_hex = hex(xor_result)[2:]
    return xor_hex.zfill(len(hex_str1))

KEY2=xor_hex_strings(KEY1,KEY21)
KEY3=xor_hex_strings(KEY23,KEY2)

FLAG=xor_hex_strings(xor_hex_strings(RES,KEY1),xor_hex_strings(KEY2,KEY3))
print(FLAG)

ANS=bytes.fromhex(FLAG)

print(ANS)