#date: 2022-08-25T16:53:18Z
#url: https://api.github.com/gists/4540fb821d21adee5df29609aec131bb
#owner: https://api.github.com/users/jermspeaks

# A "first-in-last-out" data structure
class QueueEmptyError(IndexError):
    """Attempt to pop an empty stack."""

class Queue:
    def __init__(self):
        self._list = []

    def enqueue(item):
        # Add an item to the back of the queue
        self._list.insert(0,item)

    def dequeue():
        # Remove an item from the front of the queue
        if not self._list:
            raise StackEmptyError
        else:
            self._list.pop()

    def peek():
        # Return first item of the queue
        self

#   isEmpty() {
#     // Returns a boolean if the queue is empty
#   }

#   get length() {
#     // Return the length of the queue
#   }
# }

# const queue = new Queue();
# console.log(queue);
# console.log(queue.length);
