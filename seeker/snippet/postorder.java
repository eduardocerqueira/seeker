//date: 2025-04-16T17:04:38Z
//url: https://api.github.com/gists/3c6e7697bc0bc220c00fd6fe969c3530
//owner: https://api.github.com/users/shreyanshtiwari02


class Solution {
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> ans = new ArrayList<>();
        Stack<TreeNode> stk = new Stack<>();
        TreeNode curr = root, prev = null;

        while (curr != null || !stk.isEmpty()) {
            if (curr != null) {
                stk.push(curr);
                curr = curr.left;
            } else {
                TreeNode topNode = stk.peek();
                if(topNode.right != null && topNode.right !=prev ){
                    curr = topNode.right;
                }else{
                    ans.add(topNode.val);
                    prev = topNode;
                    stk.pop();
                }

            }
        }
        return ans;
}


}