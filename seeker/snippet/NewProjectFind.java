//date: 2022-05-10T17:00:46Z
//url: https://api.github.com/gists/5a033fbb7669093418f4cc8cc0034565
//owner: https://api.github.com/users/arc0balen0

package com.khudenko;

public class NewProjectFind {

    public static void main(String[] args) {
        int[] arr = {1,2,3};
        int value = 3;
        boolean found = false;

        for (int i = 0; i < arr.length && !found; i++) {
            if (arr[i] == value) {
                found=true;
                System.out.println("found ar index: "+i);
            }
        }
        if(!found)
            System.out.println("not found");
    }
}
