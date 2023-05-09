//date: 2023-05-09T16:41:37Z
//url: https://api.github.com/gists/01183ac1f9accdc5fe10738e9bf6b947
//owner: https://api.github.com/users/kingbaldwin-iv

import java.util.*;
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
    public ListNode mergeKLists(ListNode[] lists) {
        int in;
        ArrayList<Integer> source = new ArrayList<>();
        for(ListNode n : lists) {
            while(null!=n) {
                in = Collections.binarySearch(source,n.val,(a,b)->a-b);
                in = in>=0 ? in : (-in-1);
                source.add(in,n.val);
                n = n.next;
            }
        }
        if(source.size()==0) return null;
        ListNode re = new ListNode(source.get(source.size()-1));
        for(int i = source.size()-2; i>=0; i--){
            ListNode temp = new ListNode(source.get(i),re);
            re=temp;
        }
        return re;
    }
}