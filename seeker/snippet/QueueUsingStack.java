//date: 2022-02-25T17:11:46Z
//url: https://api.github.com/gists/3f0faea935662d7909c9c1b4e70c51d4
//owner: https://api.github.com/users/itsZed0

package com.coding;

import java.util.Stack;

public class QueueUsingStack {
    public static void main(String arg[]) {
        Queue q = new Queue();
        q.add(1);
        q.add(2);
        q.add(3);
        System.out.println(q.remove());
        System.out.println(q.remove());
        System.out.println(q.remove());
    }
}

class Queue {
    Stack<Integer> stack = new Stack<>();

    public void add(int i) {
        stack.add(i);
    }

    public int remove() {
        if(stack.isEmpty()) {
            System.out.println("Queue empty");
            return -1;
        }
        else if(stack.size() ==1) {
            return(stack.pop());
        }
        else {
            int tmp = stack.pop();
            int x = remove();
            stack.push(tmp);
            return x;
        }
    }
}
