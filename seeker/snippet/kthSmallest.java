//date: 2022-02-24T16:53:38Z
//url: https://api.github.com/gists/481963762745b796723091891b65d547
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
    List<Integer> ans  = new ArrayList<>();
    public int kthSmallest(TreeNode root, int k) {
        inorder(root);
        
        return(ans.get(k-1));
    }
    
    public void inorder(TreeNode root)
    {
        if (root == null)
            return;
        
        inorder(root.left);
        ans.add(root.val);
        inorder(root.right);
    }
}