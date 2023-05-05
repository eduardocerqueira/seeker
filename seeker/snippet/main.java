//date: 2023-05-05T17:00:24Z
//url: https://api.github.com/gists/cfe7429abd74bb0513383e3b43aed959
//owner: https://api.github.com/users/niranjan-exe

#code for https://niranjan.blog/problem/good-subtrees
class Solution {
    // Main function to find number of good subtrees
    public static int goodSubtrees(Node root, int k) {
        if (root == null) {  // If root is null, return 0
            return 0;
        }
        int[] count = new int[1];  // Initialize count of good subtrees to 0
        Set<Integer> set = new HashSet<>();  // Initialize set to keep track of node values in a subtree
        countGoodSubtrees(root, k, set, count);  // Call helper function to find good subtrees
        return count[0];  // Return count of good subtrees
    }

    // Helper function to find good subtrees recursively
    private static Set<Integer> countGoodSubtrees(Node node, int k, Set<Integer> parentSet, int[] count) {
        Set<Integer> set = new HashSet<>();  // Initialize set to keep track of node values in a subtree
        if (node == null) {  // If node is null, return empty set
            return set;
        }
        // Recursively call helper function on left and right subtrees
        Set<Integer> leftSet = countGoodSubtrees(node.left, k, set, count);
        Set<Integer> rightSet = countGoodSubtrees(node.right, k, set, count);
        set.addAll(leftSet);  // Add all node values in left subtree to set
        set.addAll(rightSet);  // Add all node values in right subtree to set
        set.add(node.data);  // Add node value to set
        if (set.size() <= k && !parentSet.contains(node.data)) {  // If set size is less than or equal to k and node value is not in parent set
            count[0]++;  // Increment count of good subtrees
        }
        return set;  // Return set of node values in current subtree
    }
}
