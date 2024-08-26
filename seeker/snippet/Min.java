//date: 2024-08-26T16:46:54Z
//url: https://api.github.com/gists/415476a667aad137ad41f7de8877aae7
//owner: https://api.github.com/users/RamshaMohammed

public class Min {
    public static void main(String[] args) {
        int[] arr = {12,3,7,19,8};
        int[] r = Min(arr);
        System.out.println("Min1: " + r[0]);
        System.out.println("Min2: " + r[1]);
    }

    public static int[] Min(int[] arr) {
        int min1 = Integer.MAX_VALUE;
        int min2 = Integer.MAX_VALUE;

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] < min1) {
                min2 = min1;  
                min1 = arr[i]; 
            } else if (arr[i] < min2) {
                min2 = arr[i]; 
            }
        }
        return new int[]{min1,min2};
   }
}