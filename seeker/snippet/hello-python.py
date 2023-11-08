#date: 2023-11-08T16:41:52Z
#url: https://api.github.com/gists/0c8fa20af497589aba43b70abd9f2ad0
#owner: https://api.github.com/users/bienenbohnenbuhnen

# Lesson 1
print("Hello world")
# Lesson 2
print("   /|")
print("  / |")
print(" /  |")
print("/___|")
# Lesson 3
name = input("What is your name? ")
age = int(input("How old are you? "))
toHundred = 100 - age
print(f"{name} you wil be 100 years old in {toHundred} years")
# Lesson 4 - Odd or Even + Multiple of Four Bonus
number = int(input("Please enter a number: "))
if number % 4 == 0:
    print(f"{number} is even and a multiple of four")
elif number % 2 == 0:
    print(f"{number} is even")
elif number % 2 != 0:
    print(f"{number} is odd")
