//date: 2023-07-18T17:06:57Z
//url: https://api.github.com/gists/2531bc5dd7cb9f369c9a1a0dbcde2d03
//owner: https://api.github.com/users/yash8917

import java.util.ArrayDeque;
import java.util.Deque;

public class Demo7 {

    public static void main(String[] args) {

        Deque<String> deque = new ArrayDeque<>();    // we use ArrayList coz Deque is the interface implements the ArrayDeque
//        offers is used to add the specific element into the deque
        deque.offer("mark");
        deque.offer("John");

//        add is used to add the specific element into the deque and returns true if the operation performs the successful
        deque.add("watson");
        deque.add("Harry");
        deque.poll(); // used to remove the ele from the deque

        deque.forEach( o -> System.out.println(o));

    }
}
