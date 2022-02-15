//date: 2022-02-15T17:06:09Z
//url: https://api.github.com/gists/ec45b28cfe56033be8afb0d571e7ffbe
//owner: https://api.github.com/users/JinningYang

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
class BSTIterator {
    //InOrder Traversal
    Stack<TreeNode> stack;
    
    public BSTIterator(TreeNode root) {
        this.stack = new Stack<>();
        pushAllLeft(root);
    }
    
    public int next() {
        TreeNode node = stack.pop();
        pushAllLeft(node.right);
        return node.val;
    }
    
    public boolean hasNext() {
        return !stack.isEmpty();
    }
    
    private void pushAllLeft(TreeNode root) {
        while(root != null) {
            stack.push(root);
            root = root.left;   
        }
    }
}

/**
 * Your BSTIterator object will be instantiated and called as such:
 * BSTIterator obj = new BSTIterator(root);
 * int param_1 = obj.next();
 * boolean param_2 = obj.hasNext();
 */