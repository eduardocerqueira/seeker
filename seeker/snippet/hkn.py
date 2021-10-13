#date: 2021-10-13T17:00:43Z
#url: https://api.github.com/gists/7f4beb6243d1e2c0995a1c3842457a3f
#owner: https://api.github.com/users/norandomtechie

# HKN Git Workshop Python Demo
# Author: HKN

import hashlib
from random import *

# Used to create a grocery store.
class Inventory:
    def __init__(self, items=[]):
        self.items = items
    
    def __str__(self):
        lines = ["Items currently in inventory: "]
        for item in self.items:
            id = item.getID()
            name = item.getName()
            color = item.getColor()
            count = item.getCount()
            lines += ["{0}: name = {1}, color = {2}, count = {3}".format(id, name, color, count)]
        return '\n'.join(lines)
    
    def add(self, item):
        self.items.append(item)
    
    def rem(self, id):
        self.items = [item for item in self.items if item.id != id]
    
    def sellItem(self, sel_item, count=1):
        for item in self.items:
            if item.getID() == sel_item.getID():
                print("Transaction --> {0}, added {1}, total is {2}".format(item.getName(), count, item.getCount() - count))
                item.setCount(item.getCount() - count)
    
    def buyItem(self, sel_item, count=1):
        for item in self.items:
            if item.getID() == sel_item.getID():
                print("Transaction --> {0}, removed {1}, total is {2}".format(item.getName(), count, item.getCount() + count))
                item.setCount(item.getCount() + count)
    
    def getItems(self):
        return [[i.getName(), i.getColor(), i.getCount()] for i in self.items]

# Used to create items to add to the grocery store.
class Item:
    def __init__(self, name, color, count):
        # sets the item attributes
        self._name = name
        self._color = color
        self._count = count
        # creates a unique ID for each item, easier to reference!
        self._updateID()
    
    # getters
    def getName(self):
        return self._name
    
    def getColor(self):
        return self._color
    
    def getCount(self):
        return self._count

    def getID(self):
        return self._id
    
    # setters
    def setName(self, name):
        self._name = name
        self._updateID()
    
    def setColor(self, color):
        self._color = color
        self._updateID()
    
    def setCount(self, count):
        self._count = count
        self._updateID()
    
    # generates a unique ID for easier referencing!
    def _updateID(self):
        text = ",".join([self._name, self._color, str(self._count), str(random())])
        self._id = hashlib.sha1(text.encode()).hexdigest()

if __name__ == "__main__":
    # Create our store
    my_grocery = Inventory()
    
    # Create some items to sell
    apples = Item("Apple", "red", 10)
    oranges = Item("Orange", "orange", 20)
    hersheys1 = Item("Hershey's Milk Chocolate", "brown", 100)
    hersheys2 = Item("Hershey's Cookies & Cream", "white", 150)
    
    # Add the items to our grocery
    my_grocery.add(apples)
    my_grocery.add(oranges)
    my_grocery.add(hersheys1)
    my_grocery.add(hersheys2)
    
    # Record our transactions for the day...
    my_grocery.sellItem(apples, 10)
    my_grocery.buyItem(apples, 100)
    my_grocery.sellItem(oranges, 10)
    my_grocery.buyItem(oranges, 200)
    my_grocery.buyItem(hersheys1, 1000)
    my_grocery.sellItem(hersheys1, 500)
    my_grocery.sellItem(hersheys2, 100)
    my_grocery.buyItem(hersheys2, 1000)

    # Take a final tally before we close up!
    # for item in my_grocery.getItems():
    #     print("{0}, {1} --> {2}".format(*(item)))
    print()
    print(my_grocery)
    
    # Close up!
    exit(0)