//date: 2022-02-24T16:52:25Z
//url: https://api.github.com/gists/a0b814a8846510567d58fc039b95a57c
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
    List<Integer> ans = new ArrayList<>();
    public boolean isValidBST(TreeNode root) {
        if(root == null)
            return true;
        
        inorder(root);
        
        for(int i=1; i< ans.size(); i++)
            if(ans.get(i) <= ans.get(i-1))
                return false;
        
        return true;
    }
    
    public void inorder(TreeNode root)
    {
        if(root == null)
            return;
        
        inorder(root.left);
        ans.add(root.val);
        inorder(root.right);
        
    }
}