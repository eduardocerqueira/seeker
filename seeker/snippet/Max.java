//date: 2024-08-26T16:46:54Z
//url: https://api.github.com/gists/415476a667aad137ad41f7de8877aae7
//owner: https://api.github.com/users/RamshaMohammed

public class Max {
    public static void main(String[] args) {
        int[] arr = {12,3,7,19,8};
        int[] r = Max(arr);
        System.out.println("Max1: " + r[0]);
        System.out.println("Max2: " + r[1]);
    }

    public static int[] Max(int[] arr) {
        int max1 = Integer.MIN_VALUE;
        int max2 = Integer.MIN_VALUE;

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > max1) {
                max2 = max1;  
                max1 = arr[i]; 
            } else if (arr[i] > max2) {
                max2 = arr[i]; 
            }
        }
        return new int[]{max1,max2};
   }
}