#date: 2022-11-23T17:11:00Z
#url: https://api.github.com/gists/de8c6d6de236374146e0088a6d214154
#owner: https://api.github.com/users/claralynmarsh

# Clara's first Python program at 8 years old 11/23/22

number1 = 0
number2 = 0

def add(num1, num2):
    return num1 + num2

def subtract(num1, num2):
    return num1 - num2

def multiply(num1, num2):
    return num1 * num2

def divide(num1, num2):
    return num1 / num2

methods = {
    'add': add,
    'subtract': subtract,
    'multiply': multiply,
    'divide': divide
}

def mathRobot(method):
    print("Type the first number you want to " + method)
    number1 = int(input())

    print("Type the second number you want to " + method)
    number2 = int(input())

    print("The result is " + str(methods[method](number1, number2)))

print("Welcome to my Python calculator. Do you want to add, subtract, multiply, or divide?")
method = input()

mathRobot(method)
