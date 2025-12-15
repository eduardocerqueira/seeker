#date: 2025-12-15T17:20:01Z
#url: https://api.github.com/gists/02c648146ceda3ad6582ee8c164ce51b
#owner: https://api.github.com/users/obinexus

import sys

# Setting recursion limit for deeper trees, common in recursive AVL implementations
sys.setrecursionlimit(2000)

class Node:
    """
    Represents a node in the AVL tree.
    
    Attributes:
        key: The key used for ordering in the tree.
        value: The data associated with the key.
        height: The height of the node in the AVL tree (for balance factor calculation).
        polarity_value: The intrinsic polarity value of this node (e.g., +1 for a new 'coil packet').
        polarity_sum: The aggregated polarity state of the subtree rooted at this node.
        left: Reference to the left child node.
        right: Reference to the right child node.
    """
    def __init__(self, key, value, polarity_value=1):
        self.key = key
        self.value = value
        self.height = 1
        self.polarity_value = polarity_value # Intrinsic polarity of the inserted 'e' state
        self.polarity_sum = polarity_value  # Initially, sum is just the node's value
        self.left = None
        self.right = None

class AVLTree:
    """
    An AVL tree implementation that maintains O(log n) complexity for insertion
    and incorporates a bottom-up coil state aggregation mechanism.
    """
    def __init__(self):
        self.root = None
        self.nil_state_message = "nil nil nil ground state"

    def _get_height(self, node):
        """Returns the height of a node, or 0 if the node is None."""
        if not node:
            return 0
        return node.height

    def _get_balance(self, node):
        """Returns the balance factor of a node."""
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _resolve_coil_state(self, node):
        """
        Bottom-Up State Propagation Logic for the 'coil packet'.
        
        This method aggregates the polarity state based on the rule:
        P(N) = V(N) + P(N_L) - P(N_R)
        (e -> +e left then add -e right (conjugate/negate right-side flow))
        
        It ensures the structure's state is correctly aggregated after any tree modification,
        embodying the bottom-up parsing of state.
        """
        if not node:
            return 0 # Handles the 'nil nil nil' ground state explicitly

        # Get the already computed aggregated sums of the children
        left_sum_stored = node.left.polarity_sum if node.left else 0
        right_sum_stored = node.right.polarity_sum if node.right else 0
        
        # Aggregation Rule: Intrinsic + Left_Sum - Right_Sum (Conjugation)
        node.polarity_sum = node.polarity_value + left_sum_stored - right_sum_stored

        return node.polarity_sum


    def _right_rotate(self, y):
        """Performs a right rotation and updates heights and coil states."""
        x = y.left
        T2 = x.right

        # Perform rotation
        x.right = y
        y.left = T2

        # Update heights (bottom-up)
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        
        # Update coil state for rotated nodes (bottom-up)
        # Note: y must be resolved before x, as y is now a child of x
        self._resolve_coil_state(y)
        self._resolve_coil_state(x)

        return x

    def _left_rotate(self, x):
        """Performs a left rotation and updates heights and coil states."""
        y = x.right
        T2 = y.left

        # Perform rotation
        y.left = x
        x.right = T2

        # Update heights (bottom-up)
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))

        # Update coil state for rotated nodes (bottom-up)
        # Note: x must be resolved before y, as x is now a child of y
        self._resolve_coil_state(x)
        self._resolve_coil_state(y)

        return y

    def _insert(self, root, key, value):
        """Recursive function to insert a key-value pair into the AVL tree."""
        # 1. Standard BST Insertion
        if not root:
            # New node is inserted with intrinsic polarity_value=1
            return Node(key, value)
        
        if key < root.key:
            root.left = self._insert(root.left, key, value)
        elif key > root.key:
            root.right = self._insert(root.right, key, value)
        else:
            # Key already exists, just update value and state if needed
            root.value = value
            return root
        
        # 2. Update Height of current node (bottom-up)
        root.height = 1 + max(self._get_height(root.left), self._get_height(root.right))

        # 3. Resolve Coil State (Crucial Bottom-Up Step)
        # The node's state is aggregated immediately after its children's states are finalized
        self._resolve_coil_state(root)

        # 4. Get the balance factor
        balance = self._get_balance(root)

        # 5. Perform Rotations if needed (4 cases)
        
        # Left Left Case
        if balance > 1 and key < root.left.key:
            return self._right_rotate(root)

        # Right Right Case
        if balance < -1 and key > root.right.key:
            return self._left_rotate(root)

        # Left Right Case
        if balance > 1 and key > root.left.key:
            root.left = self._left_rotate(root.left)
            return self._right_rotate(root)

        # Right Left Case
        if balance < -1 and key < root.right.key:
            root.right = self._right_rotate(root.right)
            return self._left_rotate(root)

        return root

    def insert(self, key, value):
        """Inserts a new element and updates the tree root."""
        self.root = self._insert(self.root, key, value)
    
    def get_coil_state(self):
        """Returns the total aggregated polarity sum of the entire tree."""
        if not self.root:
            print(self.nil_state_message)
            return 0
        return self.root.polarity_sum

    def _preorder_traversal(self, root, result):
        """Helper to print node details in preorder."""
        if not root:
            return
        result.append({
            'key': root.key, 
            'value': root.value,
            'height': root.height,
            'balance': self._get_balance(root),
            'intrinsic_polarity': root.polarity_value,
            'aggregated_sum': root.polarity_sum
        })
        self._preorder_traversal(root.left, result)
        self._preorder_traversal(root.right, result)
        
    def preorder_traversal(self):
        """Returns the tree nodes and their states in preorder."""
        result = []
        self._preorder_traversal(self.root, result)
        return result

# --- Example Usage ---

if __name__ == '__main__':
    avl_tree = AVLTree()
    keys_to_insert = [10, 20, 30, 40, 50, 25]

    print("--- Initializing AVL Tree with Coil State Aggregation ---")
    print(f"Insertion sequence: {keys_to_insert}")

    for key in keys_to_insert:
        print(f"\nInserting key: {key} (Polarity +1)")
        avl_tree.insert(key, f"Data-{key}")
        
        # Print the current state of the root and the total aggregated sum
        if avl_tree.root:
            # The root is fully resolved after the insert operation
            print(f"Current Root Key: {avl_tree.root.key}, Total Coil Sum: {avl_tree.root.polarity_sum}")
        else:
            print("Tree is empty.")

    print("\n--- Final Tree State (Preorder Traversal) ---")
    print("This shows the bottom-up resolution of polarity: P(N) = V(N) + P(N_L) - P(N_R)")
    print(f"{'Key':<5}{'Value':<10}{'Height':<8}{'Balance':<9}{'Intrinsic':<12}{'Aggregated Sum':<16}")
    print("-" * 60)
    
    final_state = avl_tree.preorder_traversal()
    for node_data in final_state:
        print(
            f"{node_data['key']:<5}"
            f"{node_data['value']:<10}"
            f"{node_data['height']:<8}"