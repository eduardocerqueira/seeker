#date: 2025-11-06T17:09:36Z
#url: https://api.github.com/gists/b2a2df1fa883912364dbd0b582fac587
#owner: https://api.github.com/users/modos

import sys
import string

sys.setrecursionlimit(200000)   

def IsPalindrome(s, left=0, right=None):
    if right is None:
        right = len(s) - 1
    if left >= right:
        return True
    if s[left] != s[right]:
        return False
    return IsPalindrome(s, left + 1, right - 1)

text = input().strip()

cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())

if IsPalindrome(cleaned):
    print("YES")
else:
    print("NO")
