//date: 2022-03-25T16:50:26Z
//url: https://api.github.com/gists/4dedd50a0325db08c02da86aacb2e1aa
//owner: https://api.github.com/users/ajaypatel22

import java.util.Scanner;

public class ArrayMax {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] arr = new int[n];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = sc.nextInt();
        }
        int max = ArrMax(arr, 0);
        System.out.println(max);

    }

    public static int ArrMax(int[] arr, int idx) {
        if (idx == arr.length-1) {
            return arr[idx];
        }

        int max1 = ArrMax(arr, idx + 1);
        if (max1 > arr[idx]) {
            return max1;
        }
        else {
            return arr[idx];
        }
    }
}