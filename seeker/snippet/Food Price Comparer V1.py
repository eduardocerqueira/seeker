#date: 2021-08-31T02:13:30Z
#url: https://api.github.com/gists/305dbd54bdbab3aef29acf663a4aeb9e
#owner: https://api.github.com/users/giddycaleb

"""First Version for my assembled outcome-combine version 1 & 2
Created by Caleb Giddy
Version 1
27/08/2021
"""

# dictionary setting up common units and their matching words/symbols
unit_list = [["kg", "kilogram", "kilograms"],
             ["g", "gram", "grams"],
             ["mg", "milligram", "milligrams"],
             ["l", "litre", "litres"],
             ["ml", "millilitre", "millilitres"]]


# not_blank function which checks input is not blank and can check for numbers
def not_blank(question, error_message, num_ok):
    valid = False
    error = error_message
    while not valid:
        number = False
        response = input(question)
        if not num_ok:
            for letter in response:
                if letter.isdigit():
                    number = True

        if not response or number == True:  # Generate Error for bad name
            print(error)

        else:  # no error found
            return response

# main routine

# running not_blank function to check for numbers and blanks
product_name = not_blank("What is the name of the product? ",
                         "Your Product Name is blank or contains a digit",
                         False)
print("You are going to compare {}".format(product_name))
# running not blank function to check for numbers and blanks
units = not_blank("What unit is the product measured in? \n"
                  "Enter 'X' if product has no units e.g. 'EGGS':  ",
                  "That unit is blank or contains digits",
                  False)

# converting words to the associated symbol e.g. kilograms -> kg
for i in unit_list:
    for x in i:
        if units.lower() == x:
            units = i[0]
if units.upper() == "X":
    print("There are no units for this product")
else:
    print("Product is measured in {}".format(units))
