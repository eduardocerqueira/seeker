#date: 2025-01-15T16:53:25Z
#url: https://api.github.com/gists/9493ef71a7a9a08f169858635b8924f9
#owner: https://api.github.com/users/ssokolowskisebastian

# Data Types. Strings. Task 1
# Fractions
# Create a function that takes two parameters of string type which are fractions with the same denominator and returns a sum expression of these fractions and the sum result.

def get_fractions(a_b: str, c_b: str) -> str:
    a = a_b.split('/')
    b = c_b.split('/')
    
    return f'{a_b} + {c_b} = {int(a[0]) + int(b[0])}/{int(a[1])}'

# Implement a function get_longest_word(s: str) -> str which returns the longest word in the given string. The word can contain any symbols except whitespaces (' ', '\n', '\t' and so on). If there are multiple longest words in the string with the same length return the word that occurs first.

# Example:

# >>> get_longest_word('Python is simple and effective!')
# 'effective!'

def get_longest_word( s: str) -> str:
    words = s.split(' ')
    longest_len = 0
    
    for item in words:
        if( len(item) > longest_len ):
            longest_len = len(item)
            str = item
    return str


# Data Types. Strings. Task 3
# Implement a function that receives a string and replaces all " symbols
# with ' and vice versa.

def replacer(s: str) -> str:
    replaced = ''
    for char in range(0,len(s)):
        if(s[char]=="'"):
            replaced +='"'
        elif(s[char]=='"'):
            replaced +="'"
        else:
            replaced +=s[char]
    
    return replaced


# Data Types. Strings. Task 4
# Write a function that checks whether a string is a palindrome or not. The usage of any reversing functions is prohibited.

# To check your implementation you can use strings from here

# Examples:

# A dog! A panic in a pagoda!
# Do nine men Interpret? Nine men I nod
# T. Eliot, top bard, notes putrid tang emanating, is sad; I'd assign it a name: gnat dirt upset on drab pot toilet.
# A man, a plan, a canal — Panama!

def check_str(s: str):
    s = ''.join(x for x in s if x.isalpha()).lower()
    str= s[::-1]
    
    if(s==str):
        return True
    return False

