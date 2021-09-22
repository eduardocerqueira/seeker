#date: 2021-09-22T17:06:51Z
#url: https://api.github.com/gists/5a84c825c41c26d65e5a6bbe7eee713b
#owner: https://api.github.com/users/shilpashingnapure

'''
Maximum Length of a Concatenated String with Unique Characters

Solution
Given an array of strings arr. String s is a concatenation of a sub-sequence of arr which have unique characters.

Return the maximum possible length of s.

 

Example 1:

Input: arr = ["un","iq","ue"]
Output: 4
Explanation: All possible concatenations are "","un","iq","ue","uniq" and "ique".
Maximum length is 4.

'''
from itertools import combinations
def maxLength(arr) -> int:
    m = 0
    if len(''.join(arr)) == len(set(''.join(arr))):
        m = max(m , len(''.join(arr)))
    for i in arr:
        if len(i) == len(set(i)):
            m = max(m , len(i))    
    for i in range(2,len(arr)):
        com = list(combinations(arr,i))
        for j in com:
            val = ''.join(j)
            if len(val) == len(set(val)):
                m = max(m , len(val))
    return m 
print(maxLength(["un","iq","ue"]))