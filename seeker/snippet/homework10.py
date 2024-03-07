#date: 2024-03-07T16:54:56Z
#url: https://api.github.com/gists/13c9f4bff690dc3bbb0dddfab273e9d3
#owner: https://api.github.com/users/cesipila

# Homework Lesson 10 - Workshop - Homework

# READ CAREFULLY THE EXERCISE DESCRIPTION AND SOLVE IT RIGHT AFTER IT

################################################################################
### When solving coding challenges, think about the time complexity (Big O). ###
################################################################################

# Challenge 1
# Multiplication of a three-digit number
#
# A program gets a three-digit number and has to multiply all its digits.
# For example, if a number is 349, the code has to print the number 108, because 3*4*9 = 108.
#
# Hints:
# Use the modulus operator % to get the last digit.
# Use floor division to remove the last digit

def multiplication_of_three(number):
    # Ensure the number is three digits
    if 100 <= number <= 999:
        # convert the number to a string
        str_num = str(number)
        # Calculate the product of the digits
        product_of_digits = int(str_num[0]) * int(str_num[1]) * int(str_num[2])
        return product_of_digits
    else:
        return "Please enter a three digit number."

# Test the function
print(multiplication_of_three(123))

# ---------------------------------------------------------------------

# Challenge 2
# Sum and multiplication of even and odd numbers.
#
# You are given an array of integers. Your task is to calculate two values: the sum of
# all even numbers and the product of all odd numbers in the array. Please return these
# two values as a list [sum_even, multiplication_odd].
#
# Example
# For the array [1, 2, 3, 4]:
#
# The sum of all even numbers is 2 + 4 = 6.
# The product of all odd numbers is 1 * 3 = 3.
# The function should return the list [6, 3].

def calculate_even_odd(arr):
    sum_even = 0
    multiplication_odd = 1

    for num in arr:
        if num % 2 == 0:
            sum_even += num
        else:
            multiplication_odd *= num

    return [sum_even, multiplication_odd]

# Example usage:
my_array = [2, 3, 5, 8, 10, 7]
result = calculate_even_odd(my_array)
print(f"Sum of even numbers: {result[0]}")
print(f"Product of odd numbers: {result[1]}")


# ---------------------------------------------------------------------

# Challenge 3
# Invert a list of numbers
#
# Given a list of numbers, return the inverse of each. Each positive becomes a negative,
# and the negatives become positives.
#
# Example:
# Input: [1, 5, -2, 4]
# Output: [-1, -5, 2, -4]

def invert_numbers(numbers):
    inverted_list = []
    for num in numbers:
        inverted_list.append(-num)
    return inverted_list

# Example usage:
input_numbers = [1, 5, -2, 4]
output_numbers = invert_numbers(input_numbers)
print(f"Input: {input_numbers}")
print(f"Output: {output_numbers}")


# ---------------------------------------------------------------------

# Challenge 4
# Difference between
#
# Implement a function that returns the difference between the largest and the
# smallest value in a given list.
# The list contains positive and negative numbers. All elements are unique.
#
# Example:
# Input: [3, 5, 7, 2]
# Output: 7 - 2 = 5

def max_diff(arr):
    # Check if the list is empty
    if len(arr) == 0:
        return 0

    # Find the maximum and minimum values
    max_value = max(arr)
    min_value = min(arr)

    # Calculate the difference
    difference = max_value - min_value

    return difference

# Example usage:
input_list = [3, 5, 7, 2]
result = max_diff(input_list)
print(f"Input: {input_list}")
print(f"Output: {result}")

# If the list is not empty,
# proceed with the rest of the code.

# Your code here


# ---------------------------------------------------------------------

# Challenge 5
# Sum between range values
# You are given an array of integers and two integer values, min and max.
# Your task is to write a function that finds the sum of all elements in the
# array that fall within the range [min, max], inclusive.
#
# Example:
# arr = [3, 2, 1, 4, 10, 8, 7, 6, 9, 5]
# min_val = 3
# max_val = 7
#
# Output: 25 (3 + 4 + 5 + 6 + 7)
#
# Hint:  Iterate through each number (num) in the array (arr) and check if the current number  falls within the range [min_val, max_val].

def sum_within_range(arr, min_val, max_val):
    total_sum = 0
    for num in arr:
        if min_val <= num <= max_val:
            total_sum += num
    return total_sum

# Example usage:
arr = [3, 2, 1, 4, 10, 8, 7, 6, 9, 5]
min_val = 3
max_val = 7
result = sum_within_range(arr, min_val, max_val)
print(f"Input array: {arr}")
print(f"Range: [{min_val}, {max_val}]")
print(f"Output: {result}")
