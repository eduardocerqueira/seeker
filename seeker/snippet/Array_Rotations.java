//date: 2024-08-26T16:46:54Z
//url: https://api.github.com/gists/415476a667aad137ad41f7de8877aae7
//owner: https://api.github.com/users/RamshaMohammed

import java.util.Arrays;

class Array_Rotations {
    static void left_rotate(int[] arr, int n) {
        n = n % arr.length;
        for (int i = 0; i < n; i++) {
            int copy = arr[0];
            for (int j = 1; j < arr.length; j++) {  
                arr[j - 1] = arr[j];
            }
            arr[arr.length - 1] = copy;
            System.out.println(Arrays.toString(arr));
        }
    }

    static void right_rotate(int[] arr, int n) {
        n = n % arr.length;
        for (int i = 0; i < n; i++) {
            int copy = arr[arr.length - 1];
            for (int j = arr.length - 1; j > 0; j--) {
                arr[j] = arr[j - 1];
            }
            arr[0] = copy;
            System.out.println(Arrays.toString(arr));
        }
    }

    public static void main(String[] args) {
        int[] arr = {8, 4, 5, 1, 7, 3, 2, 4};
        System.err.println("Original array: " + Arrays.toString(arr));
        left_rotate(arr, 1);
        arr = new int[]{8, 4, 5, 1, 7, 3, 2, 4};
        right_rotate(arr, 1);
    }
}