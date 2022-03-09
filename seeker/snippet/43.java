//date: 2022-03-09T16:57:32Z
//url: https://api.github.com/gists/2ee2e66fe915ff1212d183b40ee449d1
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
    boolean flag = false;
    int sumTill = 0;
    public boolean hasPathSum(TreeNode root, int sum) {
        preOrder(root, sum);
        return flag;
    }
    
    void preOrder(TreeNode root, int sum) {
        if(root == null) return;
        sumTill += root.val;
        
        if(root.left == null && root.right == null && sumTill == sum) {
            flag = true; 
            return; 
        }
        
        preOrder(root.left, sum);
        preOrder(root.right, sum);
        sumTill -= root.val;
    }
}