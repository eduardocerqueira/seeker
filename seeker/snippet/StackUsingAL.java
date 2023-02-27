//date: 2023-02-27T17:08:39Z
//url: https://api.github.com/gists/2df50757e241e761a6e6dbf73977bd87
//owner: https://api.github.com/users/sobanjawaid26

import java.util.ArrayList;

public class StackUsingAL {

    static class Stack {
        static ArrayList<Integer> list = new  ArrayList<>();
        public static boolean isEmpty(){
            return list.isEmpty();
        }

        public static void push(int data){
            list.add(data);
        }

        public static int pop(){
            if(isEmpty())
                return -1;
            int top = list.get(list.size() - 1);
            list.remove(list.size() - 1);
            return top;
        }

        public static int peek(){
            if (isEmpty())
                return -1;
            return list.get(list.size() - 1);
        }
    }

    public static void main(String[] args) {
        Stack stack = new Stack();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        stack.push(4);

        while (!stack.isEmpty()){
            System.out.println(stack.peek());
            stack.pop();
        }
    }
}