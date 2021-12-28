#date: 2021-12-28T16:47:59Z
#url: https://api.github.com/gists/f44f76e25f455da2f30923bee3cbb230
#owner: https://api.github.com/users/tseng1026

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    # time - O(n) / space - O(n)
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def isValidBSTHelper(root: Optional[TreeNode], minima: int, maxima: int) -> bool:
            if not root: return True
            if not (minima <= root.val <= maxima): return False
            
            return isValidBSTHelper(root.left, minima, root.val - 1) and \
                   isValidBSTHelper(root.right, root.val + 1, maxima)
        return isValidBSTHelper(root, -2**31, 2**31 - 1)