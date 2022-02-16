#date: 2022-02-16T16:58:36Z
#url: https://api.github.com/gists/8a61d4b41c4d59d05524b8b2252623c4
#owner: https://api.github.com/users/PhilipDW183

class Queue:

    #create the constructor
    def __init__(self):
    
        #create an empty list as the items attribute
        self.items = []
        
    
    def enqueue(self, item):
        """
        Add item to the left of the list, returns Nothing
        Runs in linear time O(n) as we change all indices
        as we add to the left of the list
        """
        #use the insert method to insert at index 0
        self.items.insert(0, item)
        
    def dequeue(self):
        """
        Removes the first item from the queue and removes it
        Runs in constant time O(1) because we are index to
        the end of the list.
        """
        #check to see whether there are any items in the queue
        #if so then we can pop the final item
        if self.items:
            
            return self.items.pop()
        #else then we return None
        else:
            return None
          
          
    def peek(self):
        """
        Returns the final item in the Queue without removal
        
        Runs in constant time O(1) as we are only using the index
        """
        #if there are items in the Queue 
        if self.items:
            #then return the first item
            return self.items
        
        #else then return none
        else:
            return None
          
    def is_empty(self):
        """
        Returns boolean whether Queue is empty or not
        Runs in constant time O(1) as does not depend on size
        """
        return not self.items
      
      
    def size(self):
        """
        Returns the size of the stack 
        Runs in constant time O(1) as only checks size
        """
        #len will return 0 if empty
        #so don't need to worry about empty condition
        return len(self.items)
      
    def __str__(self):
        """Return a string representation of the Stack""""
        return str(self.items)