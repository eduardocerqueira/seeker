#date: 2022-10-14T17:09:00Z
#url: https://api.github.com/gists/1e97b40204e26ff45514b326c45ab397
#owner: https://api.github.com/users/arikchakma

# Assignment 01: Take a floating number as input and print it in integer.

floatingNumber = float(input("Enter a floating number: "))
print(int(floatingNumber))

# Assignment 02: Print decimal number with the fixed decimal part. (up to 2 decimal places or 3 decimal places)

sampleNumber = 2.44922343556
print("The number is: ", sampleNumber)
print("The 2 decimal number is: ", round(sampleNumber, 2))
print("The 3 decimal number is: ", round(sampleNumber, 3))


# Assignment 03: Print out your full name, your semester, your ID number, your hobby and your favorite movie name. All of the information should be in separate lines and should appear formatted like below:

fullName = input("Enter your full name: ")
semester = input("Enter your semester: ")
idNumber = input("Enter your ID number: ")
hobby = input("Enter your hobby: ")
favoriteMovie = input("Enter your favorite movie: ")

print("----------------------")
print("Full Name: ", fullName)
print("ID: ", idNumber)
print("Semester: ", semester)
print("Hobby: ", hobby)
print("Favorite Movie: ", favoriteMovie)


# Assignment 04: Take three numbers as input and short them from smallest to larget and largest to smallest.

a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
c = int(input("Enter third number: "))

if a > b and b > c:
    print("Largest to smallest: ", a, b, c)
    print("Smallest to largest: ", c, b, a)
elif a > b and b < c:
    if a > c:
        print("Largest to smallest: ", a, c, b)
        print("Smallest to largest: ", b, c, a)
    else:
        print("Largest to smallest: ", c, a, b)
        print("Smallest to largest: ", b, a, c)
elif a < b and a > c:
    print("Largest to smallest: ", b, a, c)
    print("Smallest to largest: ", c, a, b)
elif a < b and a < c:
    if b > c:
        print("Largest to smallest: ", b, c, a)
        print("Smallest to largest: ", a, c, b)
    else:
        print("Largest to smallest: ", c, b, a)
        print("Smallest to largest: ", a, b, c)


# Assignment 05: Please print the following substring from the input sting Australia.

str = "Australia"
print(str[0:3])
print(str[0] + str[2] + str[4] + str[6] + str[8])
print(str[::-1][0:4])
print((str[0] + str[2] + str[4] + str[6] + str[8])[::-1])
print(str[4:8])
print(str[0:2] + str[-3:-1])
print(str[-3:-1] + str[::-1][-2:])
