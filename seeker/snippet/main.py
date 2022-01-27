#date: 2022-01-27T17:02:55Z
#url: https://api.github.com/gists/df683a6de5849a222c70618785e2abc4
#owner: https://api.github.com/users/ALENTL

# Creating an empty list
L = []

# No of elements to be in the list
n = int(input("Enter the number of elements: "))

# For loop for number of elements to be in the list
for k in range(n):
    # Taking input from the user
    x = int(input("Enter the element: "))
    # Appending the element to the list
    L.append(x)

# Printing the list
print("The list is: ", L)

# Program Starts Here
print("\n")
print("Program Starts Here")

# Main Menu
print("\nMain Menu")

# Options
print("\n1. Copy all the even elements to a new list")
print("2. Convert all the negative numbers to positive and copy them to a new list")
print("3. Input n different elements into a list and create two new list one having all positive numbers and other having all negative numbers of the list")

# While loop for the main menu
while True:
    # Taking input from the user
    choice = int(input("Enter your choice: "))

    # Option 1
    if choice == 1:
        # Creating an empty list
        L1 = []
        # For loop for the list
        for i in L:
            # If condition for the even numbers
            if i % 2 == 0:
                # Appending the even numbers to the list
                L1.append(i)
        # Printing the list
        print("The list is: ", L1)

    # Option 2
    elif choice == 2:
        print("\nOption 2")
        # Creating an empty list
        L2 = []
        # For loop for the list
        for j in L:
            # Checking if the element is negative
            if j < 0:
                # Converting the element to positive
                j = j * -1
                # Appending the element to the list
                L2.append(j)
            else:
                # Appending the element to the list
                L2.append(j)
        # Printing the list
        print("The list is: ", L2)

    # Option 3
    elif choice == 3:
        print("\nOption 3")
        # Creating an empty list
        L3 = []
        L4 = []
        # For loop for the list
        for k in L:
            # Checking if the element is negative
            if k < 0:
                # Appending the element to the list
                L3.append(k)
            elif k > 0:
                # Appending the element to the list
                L4.append(k)
        # Printing the list
        print("The negative list is: ", L3)
        print("The positive list is: ", L4)