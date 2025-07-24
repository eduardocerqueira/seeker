#date: 2025-07-24T17:05:31Z
#url: https://api.github.com/gists/1fdbbd1b9999eb9dd283786e9e2890e1
#owner: https://api.github.com/users/varnie

import re

"""
Longest word

You are given a string s. Return the longest word from it.

A word is a continuous sequence of letters; all other characters serve as separators and are not included in the word length.

If the string contains several words of maximum length, return the first of them in the order of their appearance.

Words connected by a hyphen form a single compound word.

Example 1:

Input: s = "Большой цветок"
Output: "Большой"

Example 2:

Input: s = "Я люблю печеньки"
Output: "печеньки"

Example 3:

Input: s = "На улице сегодня солнечно, дождя нет"
Output: "солнечно"
"""

class Solution:
    def get_longest_word(self, s: str) -> str:
        chunks = re.findall(r'[\w+\-\w+|\w+]+', s)
        result = ""
        for chunk in chunks:
            if len(chunk) > len(result):
                result = chunk
            
        return result
   