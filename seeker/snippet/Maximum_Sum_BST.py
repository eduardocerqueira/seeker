#date: 2023-01-05T16:59:26Z
#url: https://api.github.com/gists/790b3c175e6617f1551cdc0c35bfbbb1
#owner: https://api.github.com/users/abhishekjkrsna

"""
Maximum Sum BST
Given a binary tree root, the task is to return the maximum sum of all keys of any sub-tree which is also a Binary Search Tree (BST).
Input Format:
The first and only line of input contains data of the nodes of the tree in level order form. The order is: data for root node, data for left child to root node,  data for right child to root node and so on and so forth for each node. The data of the nodes of the tree is separated by space. Data -1 denotes that the node doesn't exist.
Output Format:
Print the maximum sum
Sample Input 1:
1 4 3 2 4 2 5 ## Write your code here
-1 -1 -1 -1 -1 -1 4 6 -1 -1 -1 -1
Sample Output 1:
20
"""
## Write your code here
from sys import stdin, setrecursionlimit
setrecursionlimit(10**6)
import queue
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

def createleveltree(arr):
    index = 0
    if len(arr)<1 or arr[0]==-1:
        return None
    root = Node(arr[0])
    index += 1
    q = queue.Queue()
    q.put(root)
    while not q.empty():
        currentNode = q.get()
        leftChild = arr[index]
        index += 1
        if leftChild != -1:
            newNode = Node(leftChild)
            currentNode.left = newNode
            q.put(newNode)
        rightChild = arr[index]
        index += 1
        if rightChild != -1:
            newNode = Node(rightChild)
            currentNode.right = newNode
            q.put(newNode)
    return root

def isBST3(root,min_range,max_range):
    if root==None:
        return True
    if root.data<min_range or root.data>max_range:
        return False    
    isLeftWithinConstraints=isBST3(root.left,min_range,root.data-1)
    isRightWithinConstraints=isBST3(root.right,root.data,max_range)    
    return isLeftWithinConstraints and isRightWithinConstraints

def findsum(root):
    if root is None:
        return -1
    s = []
    q = queue.Queue()
    q.put(root)
    while not q.empty():
        currentNode = q.get()
        s.append(currentNode.data)
        if currentNode.left is not None:
            q.put(currentNode.left)
        if currentNode.right is not None:
            q.put(currentNode.right)
    return sum(s)

def findBSTroot(root, out):
    if root is None:
        return 
    if isBST3(root, -100000, 100000):        
        out.append(findsum(root))
    findBSTroot(root.left, out)
    findBSTroot(root.right, out)
    return
    

def maxsumbst(root):
    out = []
    findBSTroot(root, out)
    if out == []:
        print(-1)
    else:
        print(max(out))


# Main
arr = list(map(int, stdin.readline().strip().split()))
root = createleveltree(arr)
maxsumbst(root)


