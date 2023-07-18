//date: 2023-07-18T17:06:31Z
//url: https://api.github.com/gists/401feb7e3ceee2f9e339b41bab57b9f4
//owner: https://api.github.com/users/yash8917

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.PriorityQueue;

public class Demo6 {
    public static void main(String[] args) {
        PriorityQueue<String> queue = new PriorityQueue<>();
        queue.add("Mark");
        queue.add("Zobaaz");
        queue.add("John");
        queue.add("Watson");

//        has two methods 1. element() , 2. peek()
//        element() - > it is used to retrieve but , but does not remove the head of the queue and it throws the exception when the queue is Empty
//        System.out.println("Head : " + queue.element());

//      2. peek() -> it is used to retrieve but , but does not remove the head of the queue and it return the null, when the queue is Empty
        System.out.println("Head : " + queue.peek());


        queue.forEach( l -> System.out.println(l));

//        It is use to retrieve and remove the elements from the queue


//        Retrieves and removes the head of this queue, or returns null if this queue is empty.
        queue.poll();

    }
}
