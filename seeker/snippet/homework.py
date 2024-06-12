#date: 2024-06-12T17:07:20Z
#url: https://api.github.com/gists/8e6ecb8348f8ab2d417c18d45af80660
#owner: https://api.github.com/users/Karim-onward


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

my_number = -20
#make the number positive if its negtaive, and keep it.

if my_number < 0:
    my_number = - my_number
print(my_number)








# ---------------------------------------------------------------------

# Challenge 2
# BinGo!
#
# If the number is divisible of 3, print “Bin”
# If the number is divisible of 7, print “Go”
# For numbers which are divisible of 3 and 7, print “BinGo”
# Otherwise, print the original number: “{number} is just a number”



n = '21'
#123456

if n % 3 == 0 and n % 7 == 0:
    print("bingo!")
elif n % 3 == 0:
    print("bin")
elif n % 7 == 0:
    print("go")
else:
    print(f"{n} is just a number")


# ---------------------------------------------------------------------

# Challenge 3
# Find the middle number
#
# Given three different numbers x, y, and z, find the number that is neither
# the smallest nor the largest and print it.
#
# Example:
# x = 1, y = 5, z = 3 => Output: 3
x = 1
y = 5
z = 3

if x < y and z < x:
    print("x")
elif y > x and z > y:
    print("y")
elif z > x and z < y:
    print("z")
else:
    print("xyz")


# ---------------------------------------------------------------------

# Challenge 4
# Palindrome Numbers
#
# Ask a user to input a number.
# Write a program to check if the given number is a palindrome.
# It should print True if the number is a palindrome and False if it is not.
#
# Palindrome number: 121, 898

number = int("Enter a palindrome number: ")
return_number = number[::-1]

#Reverse the number
#check if the number is a plaindrome number

if number == return_number:
    print(True)
else:
    print(False)



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

Review = "tcefrep!"
Reversed_review = Review[::-1]
print(Reversed_review[1:8] + Reversed_review[0])