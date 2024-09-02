//date: 2024-09-02T16:48:04Z
//url: https://api.github.com/gists/1c69a11375b136df73dc14fb0f4c4cc9
//owner: https://api.github.com/users/sasub-mlp

import java.util.Arrays;

public class forty_three {
    public static void main(String[] args){
        int[] arr={ 5, -2, 23, 7, 87, -42, 509 };
        System.out.println("Array before reverse sort: ");
        System.out.println(Arrays.toString(arr));
        Arrays.sort(arr);
        reverse(arr);
        System.out.println("Array after reverse sort: ");
        System.out.println(Arrays.toString(arr));
    }
    static void reverse(int[] array){
        int n=array.length;
        for (int i=0;i<n/2;i++){
            int temp=array[i];
            array[i]=array[n-i-1];
            array[n-i-1]=temp;
        }
    }
}
