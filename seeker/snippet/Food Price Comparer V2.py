#date: 2021-08-31T02:14:21Z
#url: https://api.github.com/gists/595780051122c5bc48c2cfbd837a0689
#owner: https://api.github.com/users/giddycaleb

"""Second Version for my assembled outcome-combine components 1-3
Created by Caleb Giddy
Version 1
27/08/2021
"""

import re  # This is a regular Expression module

item = ""
product_list = []
price = 0

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

# Collecting Products using not_blank function
while len(product_list) < 2 or item.upper() != "X":
    item = not_blank("Please Enter Your Products in the following format\n"
                     "'Quantity' 'Name of Product/Brand' 'Price' \n"
                     "E.g. '2 Fonterra 5' for 2 Litres of Fonterra Milk for $5\n"
                     "PLEASE ENTER: ",
                     "Sorry your item cannot be blank", True)

    space_count = 0
    for i in item:  # Checking that input has right amount of spaces
        if i == " ":
            space_count += 1
    if item.upper() != "X" and 2 <= space_count:
        product_list.append(item)
    if space_count < 2 and item.upper() != "X":
        print("You inputted the data incorrectly")
    if len(product_list) < 2 and item.upper() == "X":
        # Making sure there are at least two products otherwise there is nothing to compare
        print("Sorry you need more than 1 item to compare.")

# quantity can have has mixed fraction followed by product and price
full_items_list = []
multi_word = False
product_name_list = []
product_name = ""
# The regex format below is expecting: number <space> number
mixed_regex = "\d{1,3}\s\d{1,3}\/\d{1,3}"
# \d for a digit, /d{1,3} allows 1-3 digits, /s is for space, \/ for divide

for product in product_list:
    product = product.strip()
    # Testing to see if the recipe line matches the regular expression
    if re.match(mixed_regex, product):

        # Get mixed number by matching the regex
        pre_mixed_num = re.match(mixed_regex, product)
        mixed_num = pre_mixed_num.group()
        # .group returns the part of the string where there was a match

        # Replace the space in the mixed number with '+' sign
        amount = mixed_num.replace(" ", "+")

        # Changes the string into a float using python's evaluation method
        amount = eval(amount)

        # Get unit and ingredient
        compile_regex = re.compile(mixed_regex)
        # compiles the regex into a string object - so we can search for patterns

        prod_name_price = re.split(compile_regex, product)
        # produces the recipe line unit and amount as a list

        prod_name_price = (prod_name_price[1]).strip()
        # removes the extra white space before and after the unit,
        # 2nd element in list, converting it into a string
    else:
        # splits the line at the first space
        get_amount = product.split(" ", 1)
        try:
            amount = eval(get_amount[0])  # Convert amount to float if possible
        except NameError:  # NameError rather than ValueError
            amount = get_amount[0]

        prod_name_price = get_amount[1]

    get_prod_name = prod_name_price.split(" ", 2)
    # splits the string into a list containing just the unit and ingredient

    # making it okay for there to be a space in the product/brand name
    for i in get_prod_name:
        if i[0].isdigit():
            price = i
        else:
            product_name_list.append(i)

    for i in product_name_list:
        product_name = product_name + " " + i

    product_name_list = []
    price = float(price)

    # All 3 elements of the original recipe are now broken down into the 3 variables
    list = [amount, product_name, price]
    product_name = ""

    # Adding list to a larger list containing all options
    full_items_list.append(list)

print(full_items_list)
