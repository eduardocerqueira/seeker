#date: 2023-11-10T16:58:22Z
#url: https://api.github.com/gists/5e3f41f298950640f0c6cd2e615c6f97
#owner: https://api.github.com/users/braddotcoffee

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
"""
We need to calculate the depth of the left and right
subtree all the way down. If at any point that
depth differs by > 1, we need a way to easily return
False back up the call chain.
"""
class Solution:
    def _calculate_depth(self, root) -> (int, int):
        if root is None:
            return True, 0

        left_balanced, left_depth = self._calculate_depth(root.left)
        right_balanced, right_depth = self._calculate_depth(root.right)

        if not left_balanced or not right_balanced:
            return False, 0

        if abs(left_depth - right_depth) > 1:
            return False, 0

        return True, 1 + max(left_depth, right_depth)

    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if root is None:
            return True
        
        balanced, depth = self._calculate_depth(root)
        return balanced
