#date: 2024-02-19T17:00:31Z
#url: https://api.github.com/gists/3124dba7c0dc364e88cde9390e583fbd
#owner: https://api.github.com/users/Meenaphougat

# Homework Lesson 5 - Workshop - Homework

# READ CAREFULLY THE EXERCISE DESCRIPTION AND SOLVE IT RIGHT AFTER IT

# Challenge 1
# Make a number Positive
#
# Create a variable called my_number and set it to any integer value.
# Write code to make the number positive if it's negative, and keep it
# as is if it's already positive or zero.
#
# Example:
# Input: -3 => Output: 3
# Input: 5 => Output: 5
my_number = float(input("Enter the number: "))
if my_number <= 0:
    print(my_number)
else:
    negative_number = my_number * (-1)
    print(negative_number)

# ---------------------------------------------------------------------

# Challenge 2
# BinGo!
#
# If the number is divisible of 3, print “Bin”
# If the number is divisible of 7, print “Go”
# For numbers which are divisible of 3 and 7, print “BinGo”
# Otherwise, print the original number: “{number} is just a number”
my_number = float(input("Enter the number:"))
if my_number % 21 == 0:
    print("BinGo")
elif my_number % 7 == 0:
    print("Go")
elif my_number % 3 == 0:
    print("Bin")
else:
    print(f"{my_number} is just a number")


# ---------------------------------------------------------------------

# Challenge 3
# Find the middle number
#
# Given three different numbers x, y, and z, find the number that is neither
# the smallest nor the largest and print it.
#
# Example:
# x = 1, y = 5, z = 3 => Output: 3
number_1 = float(input("Enter the first number: "))
number_2 = float(input("Enter the second number:"))
number_3 = float(input("Enter the third number:"))
if number_3 > number_1 > number_2:
    print(f"middle number is : {number_1}")
elif number_2 > number_3 > number_1:
    print(f"middle number is : {number_3}")
elif number_1 > number_2 > number_3:
    print(f"middle number is : {number_3}")
else:
    print("some numbers are equal")



# ---------------------------------------------------------------------

# Challenge 4
# Palindrome Numbers
#
# Ask a user to input a number.
# Write a program to check if the given number is a palindrome.
# It should print True if the number is a palindrome and False if it is not.
#
# Palindrome number: 121, 898
word_given = str(input("Enter your number/ number or other sequence of symbols: "))
reversed_number = str(word_given[::-1])
if reversed_number == word_given:
    print(f"{word_given} is an palindrome number")
else:
    print(f"{word_given}is not a palindrome number")

# ---------------------------------------------------------------------

# Challenge 5
# Reverse a string
#
# You're part of a team working on analyzing customer reviews for a new video game.
# Due to a software glitch, some reviews have been recorded in reverse with punctuation
# at the beginning instead of the end. Your task is to correct these reviews so that they
# are in the correct order and the punctuation is appropriately placed at the end of the
# sentence or word.
#
# Example: "tcefreP!" -> Perfect!
review_given = str(input("Enter your review:"))
if review_given[0] == "!" or review_given[0] == "." or review_given[0] == "*" or review_given[0] == "@":
    review_reversed = str(review_given[:0:-1] + review_given[0])
    print(review_reversed)
else:
    review_reversed = str(review_given[::-1])
    print(review_reversed)
