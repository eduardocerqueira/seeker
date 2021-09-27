#date: 2021-09-27T16:44:42Z
#url: https://api.github.com/gists/3e67680e6616e0fbe9579ce05aeed2e1
#owner: https://api.github.com/users/codistwa

# ============================================================
# Definition of an object / dictionary
# ============================================================

fruits = {
  'color': 'red',
  'length': 3
}

# ============================================================
# Copy by reference
# ============================================================

fruits = {
    'color': 'red',
    'length': 3
}

basket = fruits

# point to the same object
print(basket['color'])

basket['color'] = 'blue'

print(fruits['color'])  # blue

# changes
print(basket['color'])

print(basket == fruits)  # True

# ============================================================
# Delete a value key pair
# ============================================================

basket = {
    'color': 'red',
    'length': 3
}

del basket['color']
print(basket) # {'length': 3}

# ============================================================
# Add a value key pair
# ============================================================

basket = {
    'color': 'red',
    'length': 3
}

basket['size'] = 'tall'
basket.update({'size': 'tail'})
print(basket) # {'color': 'red', 'length': 3, 'size': 'tail'}