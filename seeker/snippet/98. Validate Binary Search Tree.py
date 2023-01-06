#date: 2023-01-06T16:54:11Z
#url: https://api.github.com/gists/5de31506241ac664002b2cbdea60520c
#owner: https://api.github.com/users/HsinChungHan

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class SubtreeStatus:
    def __init__(self, min_val, max_val, is_valid_bst):
        self.min_val = min_val
        self.max_val = max_val
        self.is_valid_bst = is_valid_bst

class Solution:
    def get_subtree_status(self, root)-> SubtreeStatus:
        # 需要注意當 root 為 None 時，是空子樹
        # 其 min 設定為最大值; 其 max 設定為最小值
        # 好讓上一層的 node 拿到後，作 min 和 max 的比較可以獲勝
        if root is None:
            min_val = sys.maxsize
            max_val = -sys.maxsize - 1
            return SubtreeStatus(min_val, max_val, True)
        left_status = self.get_subtree_status(root.left)
        right_status = self.get_subtree_status(root.right)
        if not left_status.is_valid_bst or not right_status.is_valid_bst:
            return SubtreeStatus(-1, -1, False)
        if left_status.max_val < root.val and right_status.min_val > root.val:
            # 這邊不能直接拿 left_status.min_val 和 right_status.max_val 
            # 當作這個 subtree 的 SubtreeStatus 的 min 和 max
            # 因為在 leaft node 的時候，我們會設定其 SubtreeStatus 的 min 和 max 為極值
            # 所以要用比較的方式找 min 和 max
            min_val = min(left_status.min_val, root.val)
            max_val = max(right_status.max_val, root.val)
            return SubtreeStatus(min_val,max_val, True)
        return SubtreeStatus(-1,-1,False)
        
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        status = self.get_subtree_status(root)
        return status.is_valid_bst




