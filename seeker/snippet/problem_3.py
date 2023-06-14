#date: 2023-06-14T17:04:04Z
#url: https://api.github.com/gists/e876d6830446d4578780881ccb06e7c7
#owner: https://api.github.com/users/aninda052

'''

PROBLEM 3
Given an integer num, repeatedly add all its digits until the result has only one digit, and return it.

Example 1:
Input: num = 38
Output: 2
Explanation: The process is
38 --> 3 + 8 --> 11
11 --> 1 + 1 --> 2
Since 2 has only one digit, return it.

Example 2:
Input: num = 0
Output: 0


'''



def convert_to_one_digit(input_number):
    
    _sum = input_number
    
    while _sum > 10:
        _sum = 0
        
        while input_number >9:

            _sum += input_number%10
            input_number = int(input_number/10)
                
        _sum += input_number
        input_number = _sum

           
    
    return _sum
  
  
convert_to_one_digit(859715)