//date: 2024-04-12T17:00:23Z
//url: https://api.github.com/gists/344b0118d7b70a3ab2e7b62dd29d97b4
//owner: https://api.github.com/users/samruds1

/**
The naive approach to flatten a binary tree into a linked list is to perform a preorder traversal of the 
tree and store the visited nodes in a Queue. 
After the traversal, start dequeuing the nodes and set the pointers of each node such that: 
the right pointer of the dequeued node is set to the previously dequeued node, and the left pointer is set to NULL.
**/

TreeNode {
 T data;
 TreeNode<T> left;
 TreeNode<T> right;
  
 TreeNode(T data) {
    this.data = data;
   this.left = null;
   this.right = null;
 }  
}

import java.util.*;

class BinaryTree<T> {
 
  TreeNode<T> root;
  
  BinaryTree(List<TreeNode<T>> ListOfNodes) {
    root = createBinaryTree(ListOfNodes); 
  }
  
  private TreeNode<T> createBinaryTree(List<TreeNode<T>> ListOfNodes) {
    if (ListOfNodes.isEmpty()) {
      return null;
    }
    
    // Create the root node of the binary tree
    TreeNode<T> root = new TreeNode<>(ListOfNodes.get(0).data);
    
    // Create a queue and add the root node to it
    Queue<TreeNode<T>> q = new LinkedList<>();
    q.add(root);
    
    // Start iterating over the listOfNodes starting from the second Node
    int i = 1;
    while (i<ListOfNodes.size()) {
     // Get the next node from the queue
      TreeNode<T> curr = q.remove();
      
      // If the node is not null, create a new TreeNode object for its left child
      // set it as the left child of the current node, and add it to the queue
      if (ListOfNodes.get(i) !=null) {
       curr.left = new TreeNode<>(ListOfNodes.get(i).data);
        q.add(curr.left);
      }
      
      i++;
      
      // If there are more ListOfNodes in this list and the next node is not null,
      // create a new TreeNode object for its right child, set it as the right child of the current node,
      // and add it to the queue. 
      if (i < ListOfNodes.size() && ListOfNodes.get(i) !=null) {
       curr.right = new TreeNode<>(ListOfNodes.get(i).data);
        q.add(curr.right);
      }
      
      i++;
    }
    
    // Return the root of the binary tree
    return root;
  }
  
 public class Main {
    public static TreeNode<Integer> flattenTree(TreeNode<Integer> root) {
        
        if (root ==null) {
          return null; 
        }
      
      // Assign current to root
      TreeNode<Integer> current = root;
      TreeNode<Integer> last = null;
      
      while (current != null) {
       
        if (current.left !=null) {
         last = current.left; 
        }
        
        while (last.right !=null) {
         last = last.right; 
        }
      }
      current = current.right;
    }
    return root;   
 }
  
} 