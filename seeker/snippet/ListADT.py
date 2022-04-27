#date: 2022-04-27T17:15:30Z
#url: https://api.github.com/gists/83aa0d2cca33bd2a8e8c3ade297e72b9
#owner: https://api.github.com/users/Professor-Sathish

#List ADT implementation

class ListADT:
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == [] 
    def add(self, item):
        self.items.append(item)
    def remove(self, item):
        self.items.remove(item)
    def size(self):
        return len(self.items)
    def display(self):
        print(self.items)
  

l=ListADT()
print("is Empty --> ",l.isEmpty())
l.add(1)
l.display()
l.add(2)
l.display()
print("List size --> ",l.size())
l.remove(1)
l.display()