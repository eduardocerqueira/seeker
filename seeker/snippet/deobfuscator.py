#date: 2023-04-27T17:08:01Z
#url: https://api.github.com/gists/18f37618865a706511b3851476eec474
#owner: https://api.github.com/users/GuyNachshon

def deobfuscator(A_0: str, A_1: str, A_2: str) -> str: 
 text = A_1 + A_2 
 text2 = "" 
 for j in range(len(A_0)): 
 text2 += chr(ord(A_0[j]) ^ ord(text[j % len(text)])) 
 return text2 