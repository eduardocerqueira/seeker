#date: 2025-01-15T17:09:14Z
#url: https://api.github.com/gists/a519a4040e9fd9f19d9157c048bc454f
#owner: https://api.github.com/users/ssokolowskisebastian

# Data Types. Lists. Task 1
# Write a Python program that accepts a sequence of words as input and prints the unique words in a sorted form.

from typing import List, Tuple

def sort_unique_elements(str_list: Tuple[str]) -> List[str]:
    
    return sorted(set(str_list))



# Data Types. Lists. Task 2
# Update the get_fizzbuzz_list function. The function has to generate and return a list with numbers from 1 to n inclusive where the n is passed as a parameter to the function. But if the number is divided by 3 the function puts a Fizz word into the list, and if the number is divided by 5 the function puts a Buzz word into the list. If the number is divided by both 3 and 5 the function puts FizzBuzz into the list.

from typing import Union, List

ListType = List[Union[int, str]]


def get_fizzbuzz_list(n: int) -> ListType:
    fizzbuzz = []    
    for i in range(1,n+1):
        if(i%5==0 and i%3==0):
            fizzbuzz.append("FizzBuzz")
            
        elif(i%3==0):
            fizzbuzz.append("Fizz")
            
        elif(i%5==0):
            fizzbuzz.append("Buzz") 
            
        else:
            fizzbuzz.append(i)
            
            
    return fizzbuzz


# Data Types. Lists. Task 3
# Implement a function foo(List[int]) -> List[int] which, given a list of integers, returns a new list such that each element at index i of the new list is the product of all the numbers in the original array except the one at i.

from typing import List


def foo(nums: List[int]) -> List[int]:
    num = []
    for i in range(0,len(nums)):
        result = 1
        for j in range(0,len(nums)):
            if(i!=j):
                result = result * nums[j]
        num.append(result)
    return num


