#date: 2022-08-25T16:53:18Z
#url: https://api.github.com/gists/4540fb821d21adee5df29609aec131bb
#owner: https://api.github.com/users/jermspeaks

# Node for LinkedList
class Node:
  def __init__(self, value):
    self.value = value
    self.next = null

# A data structure that is a series of nodes,
# and each node points to the next node of the list

# Linked lists use the “last-in-first-out” method (similar to a stack)where nodes are added to and deleted from the same end.
class LinkedList:
    def __init__(self):
        self.head = null
        self.tail = null
        self.length = 0

    def isEmpty():
        # Check whether the queue is empty
        self.length == 0

    def push(value):
        # Add an element to the linked list
        new_node = Node(value)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node 
        self.length += 1        

    def pop():
        self
        # Remove an element from the linked list  

  

    def get(index): 
        # takes an index and returns the node at that index
        self

    def delete(index): 
        # takes an index and removes node at that index
        self

    def print():
        currNode = self.head
        while (currNode != None):
            currNode = currNode.next

# list = new LinkedList()
# list.push("Emma");
# list.push("Sarah");
# list.push("Ivy");
