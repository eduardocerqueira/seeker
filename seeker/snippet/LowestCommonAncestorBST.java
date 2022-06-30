//date: 2022-06-30T16:57:42Z
//url: https://api.github.com/gists/207ed2e9131dd728e77853ed45cd751e
//owner: https://api.github.com/users/justinbaskaran

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */

class Solution {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
     /*
        Use recursion, to find the two nodes are lowest common ancestor
     */
        
      if (root == null){return null;}  
        
      if(p.val < root.val && q.val < root.val) {
          return lowestCommonAncestor(root.left,p,q);
      } else if (p.val > root.val && q.val > root.val) {
          return lowestCommonAncestor(root.right,p,q);
      } else {
          return root;
      }
        

    }
}