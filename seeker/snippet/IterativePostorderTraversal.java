//date: 2023-03-24T17:01:39Z
//url: https://api.github.com/gists/174a4fc2bb205b45f883f9f5bb23e369
//owner: https://api.github.com/users/guneyizol

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
public class IterativePostorderTraversal {
    public List<Integer> postorderTraversal(TreeNode root) {
        ArrayList<Integer> result = new ArrayList();
        ArrayDeque<TreeNode> stack = new ArrayDeque();

        TreeNode lastSeen = null;
        while (root != null || !stack.isEmpty()) {
            if (root != null) {
                stack.addLast(root);
                root = root.left;
            } else {  // if here, stack is not empty, so node is not null
                TreeNode node = stack.peekLast();

                // left subtree is done, continue with the right subtree
                if (node.right != null && node.right != lastSeen) {
                    root = node.right;
                } else {  // either there is no right subtree, or processing of the right subtree is done
                    result.add(node.val);
                    lastSeen = stack.pollLast();
                }
            }
        }

        return result;
    }
}