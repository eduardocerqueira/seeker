#date: 2025-11-17T16:56:12Z
#url: https://api.github.com/gists/48ab3ea58ca9edc31c1650bd1167c0a2
#owner: https://api.github.com/users/mattsebastianh

tables = {
  1: {
    'name': 'Chioma',
    'vip_status': False,
    'order': {
      'drinks': 'Orange Juice, Apple Juice',
      'food_items': 'Pancakes'
    }
  },
  2: {},
  3: {},
  4: {},
  5: {},
  6: {},
  7: {},
}
print(tables)


# Write your code below: 
def assign_food_items(**order_items):
  print(order_items)
  food = order_items.get('food')
  drinks = order_items.get('drinks')
  print(food)
  print(drinks)

# Example Call
assign_food_items(food='Pancakes, Poached Egg', drinks='Water')