//date: 2024-08-30T16:55:29Z
//url: https://api.github.com/gists/6efae7ebf845724b89a20348dc9521fc
//owner: https://api.github.com/users/sasub-mlp

import java.util.Scanner;

public class forty {
    static int arrsearch(int[] arr, int x) {
        for (int i=0;i<arr.length;i++){
            if (arr[i]==x){
                return i;
            }
        }
        return -1;
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int[] arr = {5, 1, 1, 9, 7, 2, 6, 10};
        System.out.print("Enter the key to find in the array: ");
        int key = sc.nextInt();
        int result=arrsearch(arr,key);
        if (result!=-1){
            System.out.println(key+" found at index "+result);
        }
        else {
            System.out.println(key+" not found in the array.");
        }
        sc.close();
    }
}
