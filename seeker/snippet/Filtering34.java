//date: 2023-01-13T16:48:29Z
//url: https://api.github.com/gists/c917a85d8bf23bd130dd41dd318dcb7a
//owner: https://api.github.com/users/ChicagoDev

import edu.princeton.cs.algs4.In;
import edu.princeton.cs.algs4.Quick;

public class Filtering34 {

    public static void minMaxFilter() {
        In in = new In();

        int max = in.readInt();
        int min = max;

        while (in.hasNextLine()) {
            int next = in.readInt();
            if (next > max) {
                max = next;
            }
            else if (next < min) {
                min = next;
            }
        }



        System.out.println("The maximum number is " + max);
        System.out.println("The minimum number is " + min);
    }

    public static void medianFilter() {
        In in = new In();
        Integer[] numsComp = new Integer[100];
        int[] nums = in.readAllInts();

        for (int i = 0; i < 100; i++)
            numsComp[i] = Integer.valueOf(nums[i]);

        Quick.sort(numsComp);

        int median = numsComp[50];

        System.out.println("The median is " + median);
    }

    public static void main(String[] args) {

        //minMaxFilter();
        medianFilter();
        
    }
}
