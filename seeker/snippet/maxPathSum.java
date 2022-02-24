//date: 2022-02-24T16:59:52Z
//url: https://api.github.com/gists/62efaf7bdb6d6b8fc1390e5c6dfbddbc
//owner: https://api.github.com/users/akshay-ravindran-96

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
    public int maxPathSum(TreeNode root) {
        
        if(root == null) return 0;
        int[] max = new int[1];
        max[0] = Integer.MIN_VALUE;
        max(root , max);
        
        return max[0];
        
    }
    
    public int max(TreeNode curr , int[] maxvalue){
        if(curr == null) return 0 ;
        
        int left = Math.max(0 , max(curr.left , maxvalue));
        int right = Math.max(0 , max(curr.right , maxvalue));
        
        maxvalue[0] = Math.max(maxvalue[0] , curr.val + left + right);
        
        return curr.val + Math.max(left , right);
    }
}