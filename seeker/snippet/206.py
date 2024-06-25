#date: 2024-06-25T16:48:53Z
#url: https://api.github.com/gists/86f60af24ba87326a731d92e4b6ad28f
#owner: https://api.github.com/users/nguyenhongson1902

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # Approach 1: Iterative solution
        # curr, prev = head, None
        # while curr:
        #     tmp = curr.next
        #     curr.next = prev
        #     prev = curr
        #     curr = tmp
        # return prev
        # Time complexity: O(n), n = len(list)
        # Space complexity: O(1)
        
        # Approach 2: Recursion
        def reverse(curr, prev):
            # base case
            if not curr:
                return prev
            
            next_node = curr.next
            curr.next = prev
            return reverse(next_node, curr)

        return reverse(head, None)
        # Time complexity: O(n)
        # Space complexity: O(1)