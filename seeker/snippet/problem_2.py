#date: 2023-06-14T17:04:04Z
#url: https://api.github.com/gists/e876d6830446d4578780881ccb06e7c7
#owner: https://api.github.com/users/aninda052

'''

PROBLEM 2

Given a string s, reverse the string according to the following rules:

All the characters that are not English letters remain in the same position.
All the English letters (lowercase or uppercase) should be reversed.
Return s after reversing it.

Example 1:
Input: s = "ab-cd"
Output: "dc-ba"

Example 2:
Input: s = "a-bC-dEf-ghIj"
Output: "j-Ih-gfE-dCba"

Constraints:
1 <= s.length <= 100
s consists of characters with ASCII values in the range [33, 122].
s does not contain '\"' or '\\'.


'''


def reverse_string(input_string):
    
    
    output_string = ""
    
    input_length = len(input_string)
    
    for idx, char in enumerate(input_string):
        
        ascii_value = ord(char)
        
        
        
        if not (ascii_value>= 65 and ascii_value<= 90) and not (ascii_value>= 97 and ascii_value<= 122):
            output_string += char
        
        print(char, ascii_value, output_string)
        output_string += input_string[input_length-idx-1]
        
    
    return output_string
  
  
reverse_string("a-bC-dEf-ghIj")