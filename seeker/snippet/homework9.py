#date: 2023-09-22T16:59:18Z
#url: https://api.github.com/gists/0253dcd3799b92478d45e8f961d4a979
#owner: https://api.github.com/users/waterfalltx

# HOMEWORK: Functions
# Read carefully until the end before you start solving the exercises.

# Basic Function
# Define a basic function that only prints Hello. Create the definition using def and the call that executes it.
def print_hello():
    print("hello")
print_hello()


########################################################################################################################
# Basic Function with Parameters
# Define a basic function that prints a greeting taking a given name.
def greetings(name):
    print("Greeting:", name)
greetings("Tom")


########################################################################################################################
# Basic Function with Default Values
# Define a basic function that prints a greeting for a name, but if none is given, use stranger instead of the name,
# so it behaves like this:

def greeting(name=None):
    if name is None:
        print("Hello, Stranger!")
    else:
        print("Hello", name)

greeting() # Prints: Hello, stranger!
greeting('Tom') # Prints: Hello, Tom!

########################################################################################################################
# Multiple Parameters
# Define a function that takes two parameters, add them up and prints the sum.
def add(a, b=0):
    c = a + b
    print(c)


add(4,0)
add(9)

########################################################################################################################
# Parameters out of order
# Define a function that takes a first_name and a last_name and prints a full_name. Define the function and create
# the function call in such a way that first_name and last_name can be given in any order and the printed full_name
# would still be correct.

def full_name(f, l):
    full = f + ' ' + l
    print(full)

# Prints: Nelson Mandela
full_name("Nelson", "Mandela")

# Is there anything you can add to the line below, so the function also prints "Nelson Mandela"?
def full_name(f, l):

    print(f + " "+ l)
    print(l + " " + f)


# Prints: Nelson Mandela
full_name("Nelson", "Mandela")


########################################################################################################################
# Returning Values
# Define a validator function that you can use to determine if a word is longer than 8 characters.
# After creating the function, make sure to test it. Create a list of words and iterate over this
# list using a for loop.

# Tip: Validator functions return True / False which we can use in conditionals to do things like print a message.

def length_func(str):
    if len(str)>8:
        print("String is too long")
    else:
        print(str)

length_func("HelloWorld")

########################################################################################################################
# You're going to revisit some of the algorithms you've already solved. But this time, there's a twist! Your challenge
# is to solve and encapsulate each algorithm into its own Python function. This will not only help you review these
# algorithms but also give you valuable practice in defining and using functions.

# FizzBuzz
# You remember FizzBuzz, right?
# You print Fizz for multiples of 3, Buzz for multiples of 5, and FizzBuzz for multiples of both 3 and 5.

# Now, your task is to take your existing FizzBuzz code and wrap it into a function called fizzbuzz.

# Requirements:
# - Create a function named fizzbuzz that takes a single argument, number.
# - If the number is a multiple of both 3 and 5, the function should return: FizzBuzz
# - If the number is a multiple of 3, the function should return: Fizz
# - If the number is a multiple of 5, the function should return: Buzz
# - Otherwise, the function should return the number.

# Pre-code
number = 15

if number % 3 == 0 and number % 5 == 0:
    print('FizzBuzz')
elif number % 3 == 0:
    print('Fizz')
elif number % 5 == 0:
    print('Buzz')
else:
    print(number)


# Wrap it into a function
def fizzbuzz(number):
    if number % 3 == 0 and number % 5 == 0:
        print('FizzBuzz')
    elif number % 3 == 0:
        print('Fizz')
    elif number % 5 == 0:
        print('Buzz')
    else:
        print(number)

# Call the function here
fizzbuzz(15)
########################################################################################################################
# Anagram
# Your next challenge is to implement a function that checks if two given strings are anagrams of each other.
# An anagram is a word or phrase formed by rearranging the letters of a different word or phrase. For example,
# "listen" is an anagram of "silent".

# What You Need to Check
# - The two strings must have the same length.
# - The sorted form of the first string must be equal to the sorted form of the second string.

# Pre-code
# Create a function named anagram. The function should take two strings as arguments
def check_anagram(n, m):

    # Check if the lengths of the strings are equal. If not, return False.
    if (sorted(n)) == (sorted(m)):
        print("The strings are anagrams.")
    else:
        print("The strings aren't anagrams.")


# Test your function with this string
t1 = 'abcde'
t2 = 'edcba'
check_anagram(t1, t2)

# Call your function here to test

########################################################################################################################
# Find Max number
# Create a function to find the largest number in a list without using the built-in max() function.

# - Define a function called find_max that takes a list of numbers as an argument.
# - Initialize a variable result and set it to the 1st item of the list using [0]
#   - This variable will hold the largest number as we iterate through the list.
# - Loop through each number in the list.
# - Check if number > result
#   - If it is, update result with the new greater number.
# - return result

# Define your function here

# Test the function with a sample list of numbers.

# Output should be the maximum number in the list.
list = []
num = int(input("How many numbers?"))
for i in range (1, num + 1):
    n = int(input("What's next?"))
    list.append(n)
print(max(list))
########################################################################################################################
# Even/Odd Checker Function
# Your task is to write a function that determines if a given integer is even or odd. The function should
# print Even for even numbers and Odd for odd numbers.

# What You Need to Check
# - Determine whether the input number is even or odd.
# - An even number can be exactly divided by 2 without a remainder.
# - An odd number leaves a remainder of 1 when divided by 2.

# Define a function is_even_odd(number) here

# Test the function calling it using a variety of numbers
# 1, 10, 5.5, 9
def odd_even(n):

    if n % 2==0:
        print("even")
    else:
        print("odd")
odd_even(5)