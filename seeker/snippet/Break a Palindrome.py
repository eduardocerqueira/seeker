#date: 2021-09-23T16:58:56Z
#url: https://api.github.com/gists/ff9ec0a3664879825144636fad255182
#owner: https://api.github.com/users/shilpashingnapure

'''
Break a Palindrome

Solution
Given a palindromic string of lowercase English letters palindrome, replace exactly one character with any lowercase English letter so that the resulting string is not a palindrome and that it is the lexicographically smallest one possible.

Return the resulting string. If there is no way to replace a character to make it not a palindrome, return an empty string.

A string a is lexicographically smaller than a string b (of the same length) if in the first position where a and b differ, a has a character strictly smaller than the corresponding character in b. For example, "abcc" is lexicographically smaller than "abcd" because the first position they differ is at the fourth character, and 'c' is smaller than 'd'.

 

Example 1:

Input: palindrome = "abccba"
Output: "aaccba"
Explanation: There are many ways to make "abccba" not a palindrome, such as "zbccba", "aaccba", and "abacba".
Of all the ways, "aaccba" is the lexicographically smallest.

'''

def breakPalindrome(palindrome):
	if len(palindrome) == 1:
		return ""
	for i in range(len(palindrome)//2):  #half of string bz ather half is same 
		#bez wnat smllest string
		if palindrome[i] != "a":
			return  palindrome[:i] + "a" + palindrome[i+1:]

	#if there is a then end of string put replace with b
	return palindrome[:-1] + "b"

print(breakPalindrome("abccba"))