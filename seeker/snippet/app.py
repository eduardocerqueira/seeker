#date: 2022-07-15T17:18:40Z
#url: https://api.github.com/gists/3bd923522958dbd02bf1748b0736ef01
#owner: https://api.github.com/users/ElijahReyes1

num1 = float(input("Enter first number: "))
op = input("Enter operator: ")
num2 = float(input("Enter second number: "))

if op == "+":
    print(num1 + num2)
elif op == "-":
    print (num1 - num2)
elif op == "/":
    print (num1 / num2)
elif op == "*":
    print (num1 * num2)
else:
    print("Invalid operator")
    