//date: 2023-05-17T16:44:24Z
//url: https://api.github.com/gists/4a7b0edad47eb9db2689aba34858cda4
//owner: https://api.github.com/users/sshehrozali

import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        // Q.
        // Take 10 integer inputs from user and store them in
        // an array. Now, copy all the elements in
        // an another array but in reverse order.

        // 3 rooms / 3 blocks
        ArrayList<Integer> fruits = new ArrayList<>();
        fruits.add(100);
        fruits.add(200);
        fruits.add(300);

        // The array is full / overflow !!!! //
        // ---------------------------------- //

        // ------ NEW ERA ----------- //
        // fruits -> 3 -> 4 = 400

        extend(fruits);
    }

    public static ArrayList extend(ArrayList previousArray) {
        // 1. We need allocate a brand new array with +1 capacity
        // 2. We need to copy the contents of previous array inside the new array
        // 3. In the last, we just need to return the brand new array

        // 1.
        // Here we created a brand new array with +1 capacity
        // int[] prev = new int[3] (old)
        // int[] new = new int[previousArray.length + 1] (new)
        ArrayList<Integer> newArray = new ArrayList<>();

        // 2.
        for (int i = 0; i < 3; i++) {
            newArray.add((Integer) previousArray.get(i));
        }

        // 3.
        return newArray;
    }
}