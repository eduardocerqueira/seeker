#date: 2022-05-09T16:58:34Z
#url: https://api.github.com/gists/25c204fb44fcefcce539bec65548d1b6
#owner: https://api.github.com/users/mh0s41n

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return None
        
        tmp = head
        
        while tmp.next is not None:
            if tmp.val == tmp.next.val:
                tmp.next = tmp.next.next
            else:
                tmp = tmp.next
            
        return head
