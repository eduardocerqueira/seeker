#date: 2026-01-06T17:12:55Z
#url: https://api.github.com/gists/11960f3aa479463e53389ac43b37ffa2
#owner: https://api.github.com/users/iamvalson

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode(0)
        current = dummy_head
        carried_number = 0

        while l1 or l2 or carried_number:
            num1 = l1.val if l1 else 0
            num2 = l2.val if l2 else 0

            total = num1 + num2 + carried_number
            carried_number = total // 10
            digit = total % 10

            newNode = ListNode(digit)
            current.next = newNode
            current = current.next

            if l1: l1 = l1.next
            if l2: l2 = l2.next
        return dummy_head.next