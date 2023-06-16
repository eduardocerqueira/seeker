//date: 2023-06-16T16:59:44Z
//url: https://api.github.com/gists/303272bbd42555593e5cb8072c56f93c
//owner: https://api.github.com/users/youngvctr

/*
 강의 내용을 참고하여 문제를 해결하였으므로 코드를 직접 구현한 부분만 남겨뒀음. 
 설명은 주석으로 기록하였음.
*/

import java.util.*;
class DoubleNode {
  // Node 정의 클래스, 변수 및 생성자 선언
}

class CircularLinkedList {
    // head, tail 변수 선언

    // CircularLinkedList class 생성자 

    public boolean isEmpty() {
        //LinkedList가 비었는지인확인
    }
    
    public DoubleNode next(DoubleNode node, int cnt){
        if(cnt == 0) return node;
        return next(node.next, cnt-1);
    }

    public void addData(int data, Integer beforeData) {
        //데이터 추가
    }

    public void removeData(int data) {
        // 데이터 제거
    }

    public String showData(String type) {
        String out = "";

        DoubleNode curHead = this.head;
        DoubleNode curTail = this.tail;
        out = String.valueOf(curHead.data);
        if (type.equals("tail")) {
            out = String.valueOf(curTail.data);
        }

        return out;
    }
}

class Main {
    public static void main(String[] arg){
        Scanner sc = new Scanner(System.in);
        String input = sc.nextLine();
        int pLength = Integer.parseInt(input.split(" ")[0]);
        int removeIdx = Integer.parseInt(input.split(" ")[1]);
        
        CircularLinkedList myList = new CircularLinkedList(new DoubleNode(0, null, null));
        myList.removeData(0);

        for(int i=1; i<pLength+1; i++){
            myList.addData(i, null);
        }

        int tempIdx = removeIdx;
        StringBuilder sb = new StringBuilder();
        sb.append("<");

        while(!myList.isEmpty()){
            myList.head = myList.next(myList.head, removeIdx-1);
            myList.tail = myList.next(myList.tail, removeIdx-1);

            String head = String.valueOf(myList.showData("head"));
            if(myList.head == myList.tail) {
                sb.append(head);
                sb.append(">");
                break;
            }
            sb.append(head);
            sb.append(", ");
            myList.removeData(Integer.parseInt(head));
        }
        System.out.println(new String(sb));
    }
}