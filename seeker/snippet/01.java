//date: 2023-11-01T16:54:05Z
//url: https://api.github.com/gists/23e92968a282f5f926fbcd5a4be74b2c
//owner: https://api.github.com/users/smatthewenglish

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

    Integer maxFrequency = 0;
    HashMap<Integer, Integer> modeMap = new HashMap<>();

    public int[] findMode(TreeNode root) {

        Integer key = root.val;
        Integer countKey = 0;

        if(modeMap.containsKey(key)) {
            countKey = modeMap.get(key) + 1;
            modeMap.put(key, countKey);
        } else {
            countKey = 1;
            modeMap.put(key, countKey);
        }

        if(countKey > maxFrequency) {
            maxFrequency = countKey;
        }

        if(root.left != null) {
            findMode(root.left);
        }
        if (root.right != null) {
            findMode(root.right);
        }

        HashSet<Integer> resultSet = new HashSet<>();
        modeMap.forEach((keyMap, countKeyMap) -> {
            if (countKeyMap.equals(maxFrequency)) {
                resultSet.add(keyMap);
            }
        });
        int[] output = resultSet.stream().mapToInt(Integer::intValue).toArray();
        return output;
    }
}