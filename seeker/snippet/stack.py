#date: 2022-08-25T16:53:18Z
#url: https://api.github.com/gists/4540fb821d21adee5df29609aec131bb
#owner: https://api.github.com/users/jermspeaks

# A "first-in-last-out" data structure
class StackEmptyError(IndexError):
    """Attempt to pop an empty stack."""

class Stack:
    def __init__(self):
        self._list = []
  
    def pop():
        # Removes the top item from the stack
        # should we check if list is not empty? => yep
        if not self._list:
            raise StackEmptyError()
        else:
            return self._list.pop()
    
        


    def push(item):
        # Adds an item to the top of the stack
        self._list.append(item)


    def peek():
        # Return the item at the front of the queue (but do not remove it)
        if not self._list:
            return StackEmptyError
        else:
            return self._list[-1]

    # def isEmpty():
    #     # Check whether the queue is empty
    

  # get length() {
  #   // Return the length of the stack

# let books = new Stack()
# books.push("Book 1")
# books.push("Book 2")
# books.push("Book 3")
# print books.peek())
# books.pop()
# print books.peek()
# print books.length
