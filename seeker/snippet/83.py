#date: 2023-10-27T17:06:05Z
#url: https://api.github.com/gists/b3d8677b89cf9da82a02b856648ba7ac
#owner: https://api.github.com/users/jssonx

class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        slow = head
        fast = head
        while fast.next:
            fast = fast.next
            if fast.val == slow.val:
                continue
            if fast.val != slow.val:
                slow.next = fast
                slow = fast
        slow.next = None
        return head