#date: 2022-07-25T16:55:24Z
#url: https://api.github.com/gists/ea6fdbe834ef017c2e93fe322890ba92
#owner: https://api.github.com/users/lishuaieric

# exercise 1
print("Hello world")

# exercise 2
number_string = input("Please input a number: ")
number = int(number_string)

if number % 2 == 1:
    print("Odd")
else:
    print("Even")

# exercise 3
number = 1

while number != 0:
    number_string = input("Please input a number: ")
    number = int(number_string)
    if number % 2 == 1:
        print("Odd")
    elif number % 2 == 0 and number != 0:
        print("Even")
    else:
        print("Goodbye")
