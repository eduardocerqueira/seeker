//date: 2021-09-01T17:01:47Z
//url: https://api.github.com/gists/3fa8d7eca180916062660001fe7ce4b1
//owner: https://api.github.com/users/kanth1

/**
 * You are given two linked lists representing two non-negative numbers. 
 * The digits are stored in reverse order and each of their nodes contain a single digit. 
 * Add the two numbers and return it as a linked list.
 * Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
 * Output: 7 -> 0 -> 8
 */

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        if(l1 == null)
            return l2;
        if(l2 == null)
            return l1;
        ListNode head = new ListNode(0);
        ListNode cur = head;
        int advance = 0;
        while(l1 != null && l2 != null){
            int sum = l1.val + l2.val + advance;
            advance = sum / 10;
            sum = sum % 10;
            cur.next = new ListNode(sum);
            cur = cur.next;
            l1 = l1.next;
            l2 = l2.next;
        }
        if(l1 != null){
            if(advance != 0)
                cur.next = addTwoNumbers(l1, new ListNode(advance));
            else
                cur.next = l1;
        }else if(l2 != null){
            if(advance != 0)
                cur.next = addTwoNumbers(l2, new ListNode(advance));
            else
                cur.next = l2;
        }else if(advance != 0){
            cur.next = new ListNode(advance);
        }
        return head.next;
    }
}
