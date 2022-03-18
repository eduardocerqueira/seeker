#date: 2022-03-18T17:02:45Z
#url: https://api.github.com/gists/9f20f13befa56f49c61a3fb4816957d5
#owner: https://api.github.com/users/kitek83

#list_task2.py
"""Use an empty list named characters and a += augmented assignment to convert string
'Birthday' into list of characters."""
characters = []
string1 = 'Birthday'

for letter in string1:
    characters += [letter]

print(f'characters={characters}')
print()

#2nd solution
characters2 = []
string2 = 'homePage'

"""out"
characters=['B', 'i', 'r', 't', 'h', 'd', 'a', 'y']
"""

characters2 += string2          #we don't need looping through the string2 to builf the new list - for it's little strange
print(f'characters2={characters2}')

"""out:
characters2=['h', 'o', 'm', 'e', 'P', 'a', 'g', 'e']"""