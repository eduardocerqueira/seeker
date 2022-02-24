//date: 2022-02-24T17:08:08Z
//url: https://api.github.com/gists/016b753b65ea8d01cb9661ccc4540267
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
    int preIndex=0;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        HashMap<Integer,Integer> map=new HashMap<>();
        for(int i=0;i<inorder.length;i++)
            map.put(inorder[i],i);
        return build(preorder,inorder,0,preorder.length-1,map);            
    }
    
    public TreeNode build(int[] preorder,int[] inorder,int start,int end,HashMap<Integer,Integer> map){
        if(start>end) return null;
        TreeNode node=new TreeNode(preorder[preIndex++]);
        int curr_index=map.get(node.val);
        node.left=build(preorder,inorder,start,curr_index-1,map);
        node.right=build(preorder,inorder,curr_index+1,end,map);
        return node;
    }
}