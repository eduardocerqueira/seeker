#date: 2022-05-25T17:21:53Z
#url: https://api.github.com/gists/07aff5646aec51f50881bd9d27e96d18
#owner: https://api.github.com/users/oreillyross


class Node:
	def __init__(self, val):
		self.val = val
		self.next = None

a = Node('A')
b = Node('B')
c = Node('C')
d = Node('D')

a.next = b
b.next = c
c.next = d

# classic imperative solution
def printList(head):
	current = head
	while current is not None:
		print(current.val)
		current = current.next

# recursive classic traversal
def printList2(head):
	#base case
	if head is None:
		return None
	# do something with each node
	print(head.val)
	#recursion
	printList2(head.next)


# printList(a)
printList2(a)