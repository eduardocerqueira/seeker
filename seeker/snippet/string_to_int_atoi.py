#date: 2022-05-16T17:15:22Z
#url: https://api.github.com/gists/200201a786bea3b4aaebf9e574bc6834
#owner: https://api.github.com/users/AbirRazzak

# https://leetcode.com/problems/string-to-integer-atoi/

class Solution:
    def convert_char_to_int(self, digit_as_char: str) -> int | None:
        if digit_as_char == '0':
            return 0
        elif digit_as_char == '1':
            return 1
        elif digit_as_char == '2':
            return 2
        elif digit_as_char == '3':
            return 3
        elif digit_as_char == '4':
            return 4
        elif digit_as_char == '5':
            return 5
        elif digit_as_char == '6':
            return 6
        elif digit_as_char == '7':
            return 7
        elif digit_as_char == '8':
            return 8
        elif digit_as_char == '9':
            return 9
        else:
            return None
    
    def is_valid_numeric_character(self, s: str):
        return self.convert_char_to_int(s) is not None
    
    def clamp_int_value(self, i: int) -> int:
        if i < (-2)**31:
            return (-2)**31
        elif i > 2**31 - 1:
            return 2**31 - 1
        else:
            return i
        
    def convert_string_to_int(self, number_as_str: str) -> int:
        # number_as_str = '42'
        number = 0
        
        num_digits = len(number_as_str)
        for i in range(len(number_as_str)):
            current_digit = number_as_str[i]
            digit_converted = self.convert_char_to_int(current_digit)
            number += 10**(num_digits-i-1) * digit_converted
        
        return number
            
    def myAtoi(self, s: str) -> int:
        leading_whitespaces = True
        leading_sign = True
        
        sign = 1
        number_substring = ''
        for c in s:
            if c == ' ' and leading_whitespaces:
                continue
            elif c == '-' and leading_sign:
                leading_whitespaces = False
                leading_sign = False
                sign = -1
            elif c == '+' and leading_sign:
                leading_whitespaces = False
                leading_sign = False
                sign = 1
            elif self.is_valid_numeric_character(c):
                leading_whitespaces = False
                leading_sign = False
                number_substring += c
            else:
                break
                
        return self.clamp_int_value(sign * self.convert_string_to_int(number_substring))
