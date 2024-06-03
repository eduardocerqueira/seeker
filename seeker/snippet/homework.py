#date: 2024-06-03T16:37:47Z
#url: https://api.github.com/gists/f4c0226859da32658d0732c12194cc69
#owner: https://api.github.com/users/Karim-onward

# Homework Lesson 4 - Conditionals

# READ CAREFULLY THE EXERCISE DESCRIPTION AND SOLVE IT RIGHT AFTER IT

# ---------------------------------------------------------------------
# Exercise 1: Temperature Classification
# You're developing a weather application. Write a program that takes 
# a temperature in Fahrenheit as input. If the temperature is above 
# 85°F, print "Hot day ahead!".
temperature = int(input("Enter the temperature in Fahrenheit: "))
#input 90

if temperature >= 85:
    print("hot day ahead!")
else:
    print("hot day ahead!")


# ---------------------------------------------------------------------
# Exercise 2: Grade Classifier
# As a teacher, you want to automate grading. Write a program that
# takes a student's score as input and prints "Pass" if the score is
# 50 or above, otherwise print "Fail".
# Do not forget that the input() function returns a string value and
# you need to convert it so you can use the value as a number.

# <Your code here>
score = input("Enter the student's score; ")

score = int(score)

if score >=50:
    print("50")
else:
    print("fail")



# ---------------------------------------------------------------------
# Exercise 3: Scholarship Eligibility
# Your university offers scholarships based on academic performance.
# Write a program that takes a student's GPA as input. If the GPA
# is greater than or equal to 3.5, print
# "Congratulations, you're eligible for a scholarship!". If it's
# between 3.0 and 3.49, print "You're on the waiting list."
# Otherwise, print "Keep up the good work."
# Do not forget that the input() function returns a string value and
# you need to convert it so you can use the value as a number.
# The function int() converts the number to an integer, and the function
# float() converts the number to a float.

gpa = float(input("Enter your GPA: "))

if gpa >= 3.5:
    print("congratulations, you are eligible for schorlarship! ")
elif gpa == 3.0 < 3.49:
    print("you are on the waiting list.")
else:
    print("keep up the good work.")

# <Your code here>

# ---------------------------------------------------------------------
# Exercise 4: Shopping Discount
# A store is offering a discount on a product. Write a program that
# takes the original price and the discount percentage as input.
# If the discounted price is less than $50, print "Great deal!".
# Otherwise, print "Might want to wait for a better offer."
original_price = float(input("Enter product original price: "))
discount_percentage = float(input("Enter discount percentage: "))

#Calculate discount discount price
discounted_amount = original_price * (discount_percentage /100)

#Discounted price
discounted_price = original_price - discounted_amount

if discounted_price < 50:
    print("Great deal!")
else:
    print("might wait for a better offer")


# <Your code here>

# ---------------------------------------------------------------------
# Exercise 5: Movie Night Decision
# You and your friends are deciding on a movie to watch. Write a
# program that takes two movie ratings as input. If both ratings
# are above 7, print "Let's watch both!". Otherwise,
# print "Let's just pick one."

#
movie_1 = int(input("Rate your movie: "))
movie_2 = int(input("Rate your movie: "))
if movie_1 and movie_2 > 7:
    print("lets watch both")
else:
    print("lets just keep one")


# ---------------------------------------------------------------------
# Exercise 6: Restaurant Recommendation
# You're building a restaurant recommendation system. Write a program
# that takes a person's mood (happy or sad) and hunger level
# (high or low) as input. If they're happy and hungry, recommend
# a fancy restaurant. If they're sad and hungry, recommend comfort food.
# For other cases, recommend a casual dining place.

mood = input("Enter your mood (happy or sad): ")
hunger_level = input("Enter your hunger level (high or sad): ")
if mood == "happy" and hunger_level == "high":
    recommendation = "i recommend a fancy restraunt"
elif mood == "sad" and hunger_level == "high":
    recomendation = "i recommend some comfort food."

else:
    recommendation = "i recommend a casual dinning place."
    print("recommendation")



# ---------------------------------------------------------------------
# Exercise 7: Exercise 7: Tax Bracket Calculator
# You're building a tax calculation system. Write a program that
# takes a person's annual income as input. Use conditionals
# to determine their tax bracket based on the following rules:

# - If income is less than $40,000, tax rate is 10%.
# - If income is between $40,000 and $100,000 (inclusive), tax rate is 20%.
# - If income is greater than $100,000, tax rate is 30%.

# Remember that a tax rate of 10% can be represented as 10/100 or 0.1

# Print the calculated tax amount for the given income.
annual_income = float(input("Enter your annual income: "))

if annual_income < 40000:
    tax_rate = 0.1
elif 40000 <= annual_income <= 100000:
    tax_rate = 0.2
else:
    tax_rate = 0.3

#Caculated tax rate
tax_amount = annual_income * tax_rate
print(f"Your tax amount is ${tax_amount}")

# ---------------------------------------------------------------------
# Exercise 8: Ticket Pricing System
# You're working on a ticket booking system for an amusement park.
# Write a program that takes a person's age as input and determines
# their ticket price based on the following rules:
# - Children (ages 3 to 12): $10
# - Adults (ages 13 to 64): $20
# - Seniors (ages 65 and above): $15
# Print the calculated ticket price for the given age.

#

age = int(input("enter your age: "))

if 3 <= age <= 12:
    ticket_price = 10
elif 13 <= age <= 64:
    ticket_price = 20
elif age >= 65:
    ticket_price = 15
else:
    ticket_price = 0  #Assuming the ticket price is
print(f" the ticket price of an age {age} is ${ticket_price}")



# ---------------------------------------------------------------------
# Exercise 9: "**********"
# Create a program that takes a password as input and checks its
# strength based on the following rules:

# If the password is less than 8 characters, print "Weak password."
# If the password is 8 to 12 characters long, print "Moderate password."
# If the password is more than 12 characters, print "Strong password

# You can use len() function to get the length of a given string.

password = input("Enter your password: "**********"

#Determine the length of the password
password_length = "**********"

 "**********"i "**********"f "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"l "**********"e "**********"n "**********"g "**********"t "**********"h "**********"  "**********"< "**********"  "**********"8 "**********": "**********"
    print("weak password")
 "**********"e "**********"l "**********"i "**********"f "**********"  "**********"8 "**********"  "**********"< "**********"= "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"l "**********"e "**********"n "**********"g "**********"t "**********"h "**********"  "**********"< "**********"= "**********"  "**********"1 "**********"2 "**********": "**********"
    print("moderate password")
else:
    print("strong password")

