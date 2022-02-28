//date: 2022-02-28T16:57:17Z
//url: https://api.github.com/gists/cf29c057f074eb098538a6a4930c45e4
//owner: https://api.github.com/users/JinningYang

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode removeZeroSumSublists(ListNode head) {
        if(head == null) {
            return head;
        }
        
        int prefix = 0;
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        Map<Integer, ListNode> map = new HashMap<>();
        
        for(ListNode curr = dummy; curr != null; curr = curr.next) {
            prefix += curr.val;
            map.put(prefix, curr);
        }
        
        prefix = 0;
        for(ListNode curr = dummy; curr != null; curr = curr.next) {
            prefix += curr.val;
            curr.next = map.get(prefix).next;
        }
        
        return dummy.next;
    }
}