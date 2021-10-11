//date: 2021-10-11T17:09:29Z
//url: https://api.github.com/gists/fba573459ac5d7c7404e7048e707586c
//owner: https://api.github.com/users/superzjn

/**
 * Definition of TreeNode:
 * public class TreeNode {
 *     public int val;
 *     public TreeNode left, right;
 *     public TreeNode(int val) {
 *         this.val = val;
 *         this.left = this.right = null;
 *     }
 * }
 */

public class Solution {
    /**
     * @param root: the root of binary tree
     * @return: the root of the minimum subtree
     */
    TreeNode minRoot;
    int minSum;

    public TreeNode findSubtree(TreeNode root) {
        
        if (root == null) {
            return root;
        }
        minRoot = null;
        minSum = Integer.MAX_VALUE;
        getMinSumRoot(root);
        return minRoot;
    }

    public int getMinSumRoot(TreeNode root) {

        if (root == null) {
            return 0;
        }

        int sum = getMinSumRoot(root.left) + getMinSumRoot(root.right) + root.val;
        if (sum < minSum) {
            minRoot = root;
            minSum = sum;
        }

        return sum;
    }
}