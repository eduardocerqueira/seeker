#date: 2023-12-19T17:02:05Z
#url: https://api.github.com/gists/7f04edd732da89cb4db238cc1d83cbf8
#owner: https://api.github.com/users/dlagrenade

# Homework Lesson 3 - Strings

# READ CAREFULLY THE EXERCISE DESCRIPTION AND SOLVE IT RIGHT AFTER IT

# ---------------------------------------------------------------------
# Exercise 1: Personalized Greeting
# Write a program that takes a user's name as input
# and then greets them using an f-string: "Hello, [name]!"
#
# Example Input: "Alice"
# Example Output: "Hello, Alice!"
name = input("Enter yor name: ")
print(f"Hello, {name}!")

# ---------------------------------------------------------------------
# Exercise 2: Greeting with User's Favorite Activity
# Write a program that takes a user's name and their
# favorite activity as input, and then greets them
# using the formatting method of your choice as:
# "Hello, [name]! Enjoy [activity]!"

# Example Input:
# Name: Emily
# Favorite Activity: hiking
# Example Output: "Hello, Emily! Enjoy hiking!"

name = input("Greeting's, what is your name? ")
activity = input("What is your favorite hobbie? ")
print(f"Hello, {name}! Enjoy {activity}!")

# ---------------------------------------------------------------------
# Exercise 3: Membership Cards
# You are designing a simple registration system for a club.
# When new members sign up, you want to ensure their names
# are displayed in uppercase on their membership cards.
# Write a program that takes the new member's name as
# input and prints it in uppercase and prints a welcome message
# using .format()

# Example Input:
# Name: Emily
# Example Output: "Welcome, Emily! Your name in uppercase is: EMILY!"

name = input("To sign up for the membership card, please provide first name: ")
print(f"Welcome, {name.format().upper()}!")

# ---------------------------------------------------------------------
# Exercise 4: User Profile Creation
# Build a user profile generator. Ask
# the user for their first name, last name, and age. Create
# a profile summary using .title(), .upper(), and .format().
#
# Example Input:
# First name: "john"
# Last name: "smith"
# Age: 28

name_1 = input("What is your first name: ")
name_2 = input("What is your last name: ")
age = input("What is you age: ")
print("Name: {} {}".format(name_1.title(), name_2.title()))
print(f"Age: {age}")

name_3 = input("What is your first name: ")
name_4 = input("What is your last name: ")
age = input("What is you age: ")
print("Name: {} {}".format(name_3.upper(), name_4.upper()))
print(f"Age: {age}")

#
# Example Output:
# Name: John Smith
# Age: 28


# ---------------------------------------------------------------------
# Exercise 5: Text message limits
# You are developing a text messaging application that limits the
# number of characters in a single message. Your task is to create
# a Python program that takes a message as input from the user.
# The program should calculate and display the number of characters
# in the message, including spaces, and format the output using
# an f-string. This character count will help users ensure their
# messages fit within the allowed limit.
text_message = input("Hello, looking for new show to watch; can you recommend one: ")
print(f"Thank you for the recommendation, can wait to watch {text_message}!")
print(len(text_message))

# ---------------------------------------------------------------------
# Exercise 6: Text Transformation Game
# Create a text transformation game. Ask the user
# to enter a sentence. Replace all vowels with '*'. Display the
# modified sentence.
#
# Example Input: "Hello, world!"
# Example Output: "H*ll*, w*rld!"
sentence = input("Enter a sentence: ")
transformed_sentence = sentence.replace('a', '*')
print(transformed_sentence) #Welcome to the video game convention!


# ------------------------------# ---------------------------------------------------------------------
# Exercise 7: Extracting Information
# The variable 'data' is a student record in the format "name:age"
# Use string slicing and string methods to extract the name and the age
# and print the result formatted.
#
# data = "lucy smith:28"
#
# Expected output:
# Name: Lucy Smith
# Age: 28

data = "lucy smith:28"
x = data[0:10]
print(f"Name: {x.title()}")
y = data[11:]
print(f"Age: {y}")


# ---------------------------------------------------------------------
# Exercise 8: Miles to Kilometers Conversion
# Write a program that converts a distance in miles to kilometers.
# Take the distance in miles as input, convert it to kilometers
# using the formula miles * 1.6, and display the
# result using f-strings.

# Example Input: 10
# Example Output: 10 miles is approximately 16.0 kilometers.

# We are converting the input string to float:
# Input: float("1.23")
# Output: 1.23
miles = float(input("Enter distance in miles: "))


# ---------------------------------------------------------------------
# Exercise 9: Workouts calculator
# Write a Python program that asks the user to input the number
# of minutes spent on three different exercises: cardio, strength
# training, and yoga using the input() function. Convert the input
# strings to integers using the int() function. Calculate the
# total time spent on workouts by summing up the minutes from all
# three activities. Based on the total workout time, provide a
# motivational message using an f-string that encourages the user
# to stay consistent and reach their fitness goals. Display the
# motivational message to the user.
cardio = input("cardio: ")
stre = input("strength: ")
train = input("training: ")
yo = input("yoga: ")

total = int(cardio + stre + train + yo)

print(total)
