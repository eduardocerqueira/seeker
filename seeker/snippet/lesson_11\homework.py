#date: 2023-12-25T17:01:18Z
#url: https://api.github.com/gists/2c76c4f4effc8182d01761aea19ddd14
#owner: https://api.github.com/users/IrynaBulgakova

# HOMEWORK: Dictionaries
# Read carefully until the end before you start solving the exercises.

# Basic Dictionary

# Create an empty dictionary and then add a few of your friends. Make the key their email (can be fake)
# and the value their name. When you're done, create the same dictionary as a pre-populated dictionary.
friends = {}
friends["linastrat7@yahoo.com"] = "Lina"
friends["interann87@gmail.com"] = "Anna"
friends["ignatkat88@hotmail.com"] = "Kate"
print(friends)

friends = {
"linastrat7@yahoo.com" : "Lina",
"interann87@gmail.com" : "Anna",
"ignatkat88@hotmail.com": "Kate",
}
print(friends)
# ----------------------------------------------------------------------------------------------------------------------

# Nested Dictionary

# Create a nested dictionary for a list of 5 company employees.
# The key should be their employee id (an integer from 1-5 will do) and the value should be a dictionary with
# the name, department and salary of the employee.
company_employees = {
     1: {"name": "Denisa", "department": "tech_support", "salary": 35000},
     2: {"name": "Edlira", "department": "customer_service", "salary": 30000},
     3: {"name": "Jane", "department": "hr", "salary": 40000},
     4: {"name": "Dorina", "department": "management", "salary": 45000},
     5: {"name": "Kadi", "department": "rep", "salary": 41000},
}
# ----------------------------------------------------------------------------------------------------------------------

# Accessing Values

# Use the previous nested dictionary and write some print statements that will answer the following:

# - Print a list of the employee IDs
# - Print the employee data for employee with the ID 3.
# - Loop over the employees and print all their names and salaries.
employee_id = list(company_employees.keys())
print(f"Employee IDs:{employee_id}")
print(company_employees[3])
print("Employee Names and Salaries:")
for employee_id, employee_info in company_employees.items():
    print(f"{employee_info['name']}, ${employee_info['salary']}")
# ----------------------------------------------------------------------------------------------------------------------

# Updating Values

# We have the following dictionary with employee salaries.

salaries = {
    'james': 10000,
    'tom': 15000,
    'ryan': 16000,
    'julia': 17000
}

# We need to increase everyone's salary by 1,000 and also add a new employee joseph with a salary of 18,000.
# Please come up with a way to do this using update()
for employee in salaries:
    salaries[employee] += 1000

salaries.update({'joseph': 18000})

print(salaries)
# ----------------------------------------------------------------------------------------------------------------------

# Deleting Values

# You remember those employees from Updating Values section? Well, Julia got fired, so we need to remove her
# name from the salaries dictionary. How would you do that?
if 'julia' in salaries:
    del salaries['julia']

print(salaries)
# ----------------------------------------------------------------------------------------------------------------------

# Iterating over Dictionaries

# Given the list of movies below, please use view objects (keys(), values(), items() - where necessary) to answer
# the questions:

# - Is Black Panther in the list of movies?
# - Is there any movie for the year 2021?
# - Print a message for each element that shows the year, the title and the position in the dictionary (1-5).
#   Hint: use a counter.

films = {
   2016: "Star Wars: Episode VII - The Force Awakens",
   2017: "Star Wars: Episode VIII - The Last Jedi",
   2018: "Black Panther",
   2019: "Avengers: Endgame",
   2020: "Bad Boys for Life"
}
if "Black Panther" in films.values():
    print("'Black Panther' is in the list of movies.")
if 2021 in films.keys():
    print("Yes, there is a movie for the year 2021")
else:
    print("There's no movie for year 2021")
for index, (year, title) in enumerate(films.items(), start=1):

    print(f"{index}.Year:{year}, Title:{title}")
# ----------------------------------------------------------------------------------------------------------------------

# Exercise 1. Animal Shelter Volunteer

# You volunteer in a local animal shelter, and you've stored information
# about different pets in a Python dictionary.
# Your task is to create a Python program that helps you:

# - Access the age of specific pets by their names.
# - Find out which pets are not yet adopted.
# - Identify the most common animal type in the shelter (e.g., dog, cat).

# Pre-code
# Initialize the shelter_pets dictionary
shelter_pets = {
  'Whiskers': {'Age': 2, 'Type': 'Cat', 'Adopted': False},
  'Fido': {'Age': 4, 'Type': 'Dog', 'Adopted': True},
  'Patch': {'Age': 1, 'Type': 'Dog', 'Adopted': False},
  'Snowball': {'Age': 3, 'Type': 'Rabbit', 'Adopted': True}
}

# Access and print the age of Whiskers
print(shelter_pets['Whiskers']['Age'])  # Should output 2

# Access and print if Patch is a dog or cat
print(shelter_pets['Patch']['Type'])

# Access and print if Snowball is adopted or not
print(shelter_pets['Snowball']['Adopted'])

# Find out which pets are not yet adopted and print their names
for pet, info in shelter_pets.items():
    if not info['Adopted']:
        print(pet)

# ----------------------------------------------------------------------------------------------------------------------

# Exercise 2. Best-selling books

# Create a Python program to manage a collection of best-selling books
# and their publication years. Your initial list of best-selling books may have inaccuracies
# or could be outdated. Therefore, you'll also practice updating single and multiple entries
# in your dictionary. Specifically, you will:

# - Update the title of a book for a specific year due to a naming convention change.
# - Update the titles for multiple books at once based on new sales data.
# - Delete a book and its year from the collection as it's no longer considered a bestseller.
# - Print the title of the book published in a specific year.

# Pre-code:
# Initialize a dictionary called best_selling_books to store your collection.

best_selling_books = {
  1997: "Harry Potter and the Philosopher's Stone",
  1984: "Neuromancer",
  2003: "The Kite Runner",
  2015: "Go Set a Watchman"
}

# The U.S. title for the Harry Potter book published in 1997 is "Harry Potter and the Sorcerer's Stone".
# Update the title to its U.S. version.
best_selling_books[1997] = "Harry Potter and the Sorcerer's Stone"
best_selling_books.update()

# New sales data reveals that "The Hunt for Red October" was the actual bestseller for 1984
# and "The Da Vinci Code"  for 2003.
# Update these in a single operation.

best_selling_books.update({
  1984: "The Hunt for Red October",
  2003: "The Da Vinci Code",
})
print(best_selling_books)

# The book published in 2015, "Go Set a Watchman," is no longer considered a bestseller.
# Use the del keyword to remove this entry from the dictionary.
del best_selling_books[2015]
print(best_selling_books)
# Print the updated dictionary of best-selling books.

# ----------------------------------------------------------------------------------------------------------------------

# Exercise 3. Manage Music Collection
# Create a Python program that manages a collection of music albums and their release years.
# You will use a dictionary where the keys are the release years and the values are the names of albums.
# The program should allow you to:

# - Print all the release years (keys) from the dictionary.
# - Print all the album names (values) from the dictionary.
# - Print both the release years and album names together.
# - Check if a particular year or album exists in the collection.


# Steps and pre-code
# Initialize a dictionary to store Bob Dylan's albums
dylan_albums = {
  1962: "Bob Dylan",
  1963: "The Freewheelin' Bob Dylan",
  1975: "Blood on the Tracks",
  1997: "Time Out of Mind"
}

# Use .keys() to loop through and print out all the release years
for year in dylan_albums.keys():
    print(year)

# Use .values() to loop through and print out all the album names
for name in dylan_albums.values():
    print(name)
# Use .items() to loop through and print out both the release year and album name
for year, name in dylan_albums.items():
    print({f"Year: {year}. Album name: {name}"})
# Use the 'in' keyword to check if a particular year or album is in the dictionary (pick any year and any album)
# Remember the keyword by default checks only the keys, not the values.
# If you want to check if a particular value (in this case, an album name),
# you need to specify that you're searching within the dictionary's values.
if 1975 in dylan_albums.keys():
    print(True)
# ----------------------------------------------------------------------------------------------------------------------

# Exercise 4. Remove duplicates
# Remove duplicates from the following dictionary:
person = {
  'first': 'Jeff',
  'name': 'Jeff',
  'last': 'Smith',
  'last_name': 'Smith',
  'state': 'CA',
  'age': 55
}

# Steps:
# - Create a dict person
# - Create an empty dictionary result = {}
# - Make a for loop to iterate over person dictionary
# - If item’s value not in result dict, add key value part into result.
result = {}
for key, value in person.items():
    if value not in result.values():
        result[key] = value
print(result)
# ----------------------------------------------------------------------------------------------------------------------

# Exercise 5. Find the highest score
# Create a Python function named find_max_score that takes a dictionary of names and scores as an argument.
# The function should return the name and score of the person with the highest score in the form of a dictionary.

# Sample test_scores dictionary
# test_scores = {
#   'James': 83,
#   'Julia': 91,
#   'Ryan': 90,
#   'Maria': 80,
#   'David': 79,
#   'Adam': 96,
#   'Jennifer': 97,
#   'Susan': 77
# }

#  Find the person with the highest test score and display both their name and score

# Steps:
# - Define a function called find_max_score that takes one argument, scores_dict, which is a dictionary of names
#   (as keys) and scores (as values).
# - Create an empty result variable
# - Assume the initial maximum score is 0
# - Iterate over each key-value pair in the test_scores, using the .items() method
# - Check if the current score (v) is >= to the current maximum score
# - If so, update the max score and assign the key-value pair to the result
# - Return result and test the function


def find_max_score(scores_dict):
    max_score = 0
    person_max_score = None

    for name, score in scores_dict.items():
        if score >= max_score:
           max_score = score
           person_max_score = name
    return {'name': person_max_score, 'score': max_score}

test_scores = {
  'James': 83,
  'Julia': 91,
  'Ryan': 90,
  'Maria': 80,
  'David': 79,
  'Adam': 96,
  'Jennifer': 97,
  'Susan': 77
}
highest_score = find_max_score(test_scores)
print(highest_score)