//date: 2022-03-09T17:02:21Z
//url: https://api.github.com/gists/dec58aafbdde00ca9725b267549cefc7
//owner: https://api.github.com/users/TehleelMir

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    TreeNode firstElement = null;
    TreeNode secondElement = null;
    TreeNode prevNode = new TreeNode(Integer.MIN_VALUE);
    public void recoverTree(TreeNode root) {
        inOrder(root);
        int temp = firstElement.val;
        firstElement.val = secondElement.val;
        secondElement.val = temp;
    }
    
    void inOrder(TreeNode root){
        if(root == null) return;
        
        inOrder(root.left);
        
        if(firstElement == null && prevNode.val > root.val) firstElement = prevNode;
        if(firstElement != null && prevNode.val > root.val) secondElement = root;
        prevNode = root;
        
        inOrder(root.right);
    }
}