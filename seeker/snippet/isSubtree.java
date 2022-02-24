//date: 2022-02-24T16:51:17Z
//url: https://api.github.com/gists/28e9ff9ae4492411a6a405fa4bc61815
//owner: https://api.github.com/users/akshay-ravindran-96

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
    public boolean isSubtree(TreeNode s, TreeNode t) {
        return Traverse(s,t);
    }
    
    public boolean Traverse(TreeNode s, TreeNode t)
    {
        
        return (s!= null) &&(isSameTree(s, t) || Traverse(s.left, t) || Traverse(s.right, t));
        
    }
    
    public boolean isSameTree(TreeNode s, TreeNode t)
    {
        if(s== null && t == null)
            return true;
        
        if(s == null || t == null)
            return false;
        
        return (s.val == t.val && isSameTree(s.left, t.left) && isSameTree(s.right, t.right));
    }
    
}