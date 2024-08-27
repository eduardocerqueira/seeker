#date: 2024-08-27T16:45:31Z
#url: https://api.github.com/gists/373dd8d79f9319eccd04c979cf2a9396
#owner: https://api.github.com/users/nahid111

from typing import List

"""
Problem 1: Add Digits
Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.

Example 1:

Input: num = 38
Output: 2
Explanation: The process is as follows: 3 + 8 = 11, then 1 + 1 = 2. Since 2 has only one digit, 2 is returned.
"""


def add_digits(num: int) -> int:
    if num == 0:
        return num

    num = str(num)

    if len(num) == 1:
        return int(num)

    res = 0
    for n in num:
        res += int(n)

    return add_digits(res)


"""
Problem 2: Contains Duplicate
Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

Example 1:

Input: nums = [1,2,3,1]
Output: true
Explanation: The value 1 appears twice in the array.
"""


def contains_duplicate(nums: List[int]) -> bool:
    if not nums:
        return False

    cache_ = {}

    for i in nums:
        if i in cache_:
            return True
        else:
            cache_[i] = 1

    return False


"""
Problem 3: Reverse Vowels of a String
Write a function that takes a string as input and reverses only the vowels of a string.

Example 1:

Input: s = "algorithm"
Output: "ilgorathm"
Explanation: The vowels "e" and "o" are reversed.
"""


def reverse_vowels(word: str):
    if word == "":
        return ""

    word = list(word)
    vowels = ['a', 'A', 'e', 'E', 'i', 'I', 'o', 'O', 'u', 'U']
    left, right = 0, len(word)-1

    while left < right:
        if word[left] in vowels:
            for c in range(right, left, -1):
                if word[c] in vowels:
                    word[left], word[c] = word[c], word[left]
                    right = c-1
                    break
        left += 1
    return "".join(word)

