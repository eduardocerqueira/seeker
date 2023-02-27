//date: 2023-02-27T17:07:32Z
//url: https://api.github.com/gists/52011012162260d91aebe085fdb3cc3c
//owner: https://api.github.com/users/sobanjawaid26

public class StackUsingLL {

    // Stack impl using Linked List
    static class Node{
        int data;
        Node next;
        public Node(int data){
            this.data = data;
            next = null;
        }
    }
    static class Stack {
        public static Node head;
        public static boolean isEmpty(){
            return head == null;
        }

        public static void push(int data){
            Node newNode = new Node(data);
            if(isEmpty()){
                head = newNode;
                return;
            }
            newNode.next = head;
            head = newNode;
        }

        public static int pop() {
            if(isEmpty())
                return -1;

            int top = head.data;
            head = head.next;
            return top;
        }

        public static int peek(){
            if(isEmpty())
                return -1;

            return head.data;
        }

        // Push at the Bottom of Stack
        public static void pushAtBottom(int data, Stack stack){
            if(stack.isEmpty()){
                stack.push(data);
                return;
            }

            int top = stack.pop();
            pushAtBottom(data, stack);
            stack.push(top);
        }
        public static void reverseStack(Stack stack){
            if(stack.isEmpty())
                return;
            int top = stack.pop();
            reverseStack(stack);
            stack.pushAtBottom(top, stack);
        }
    }

    public static void main(String[] args) {
        Stack stack = new Stack();
        stack.push(1);
        stack.push(2);
        stack.push(3);
        stack.push(4);

        System.out.println("Reverse the stack : ");
        stack.reverseStack(stack);
        while (!stack.isEmpty()){
            System.out.println(stack.peek());
            stack.pop();
        }
        System.out.println("PushAt Bottom : ");
        stack.pushAtBottom(5, stack);
        while (!stack.isEmpty()){
            System.out.println(stack.peek());
            stack.pop();
        }
    }
}
