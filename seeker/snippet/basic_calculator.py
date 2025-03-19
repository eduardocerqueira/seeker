#date: 2025-03-19T17:09:45Z
#url: https://api.github.com/gists/51e6b8b0d987b38fc34c0c35cd56e234
#owner: https://api.github.com/users/TroyMoses

def calculator():
    print("Welcome to the Basic Calculator Program! 🧮")
    print("--------------------------------------------")

    # Store the first and second number from the user
    num1 = float(input("\nEnter the first number: "))
    num2 = float(input("\nEnter the second number: "))

    # Store the operation from the user
    operation = input("\nEnter the operation (+, -, *, /): ")

    # Calculate the result based on the operation
    if operation == "+":
        result = num1 + num2
        print(f"Solution: {num1} + {num2} = {result}")
    elif operation == "-":
        result = num1 - num2
        print(f"Solution: {num1} - {num2} = {result}")
    elif operation == "*":
        result = num1 * num2
        print(f"Solution: {num1} * {num2} = {result}")
    elif operation == "/":
        if num2 == 0:
            print("Sorry: Cannot divide by zero!")
        else:
            result = num1 / num2
            print(f"Solution: {num1} / {num2} = {result}")
    else:
        print("\nInvalid operation! Please enter a valid operation +, -, *, /.")
    
    print("\n--------------------------------------------")
    print("Thanks for using the Basic Calculator Program! 😊")

# Run the program
calculator()