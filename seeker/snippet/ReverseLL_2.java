//date: 2022-07-08T17:00:32Z
//url: https://api.github.com/gists/9af8d6b30476399afd81008ab4fc426d
//owner: https://api.github.com/users/Navila48

 public ListNode reverseBetween(ListNode head, int left, int right) {
        int l=left;
        ListNode dummy=new ListNode();
        dummy.next=head;
        ListNode prevLeft=dummy,leftPointer=dummy.next;
        while(l>1){
            leftPointer=leftPointer.next;
            prevLeft=prevLeft.next;
            l--;
        }
        ListNode cur=leftPointer;
        ListNode prev=null,next=null;
        while(left<=right && cur!=null){
            next=cur.next;
            cur.next=prev;
            prev=cur;
            cur=next;
            left++;
        }
        prevLeft.next=prev;
        leftPointer.next=cur;
        return dummy.next;
    }