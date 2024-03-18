//date: 2024-03-18T16:57:22Z
//url: https://api.github.com/gists/cd0b806ae001c406e347fb78526ebbb7
//owner: https://api.github.com/users/DarlanNoetzold

class Solution {
  public ListNode partition(ListNode head, int x) {
    ListNode beforeHead = new ListNode(0);
    ListNode afterHead = new ListNode(0);
    ListNode before = beforeHead;
    ListNode after = afterHead;

    for (; head != null; head = head.next)
      if (head.val < x) {
        before.next = head;
        before = head;
      } else {
        after.next = head;
        after = head;
      }

    after.next = null;
    before.next = afterHead.next;

    return beforeHead.next;
  }
}