//date: 2022-03-09T16:57:24Z
//url: https://api.github.com/gists/4f23013941f6e0f31d1f47f518200171
//owner: https://api.github.com/users/Danush-TBR

import java.util.Scanner;

public class quickSort {
    public static int partition(int[] arr,int start,int end){
        int i=start, j=end-1, PivotIndex = end;
        while(i<j){
            while(i<end && arr[i]<=arr[PivotIndex]) i++;
            while(j>start && arr[j]>arr[PivotIndex]) j--;
            if(i<j){
                int temp = arr[i];
                arr[i]=arr[j];
                arr[j]=temp;
            }
        }
        int temp=arr[PivotIndex];
        arr[PivotIndex]=arr[i];
        arr[i]=temp;
        return i;
    }
    public static void QuickSort(int[] arr,int start,int end){
        if(start<end){
            int partitionIndex = partition(arr, start, end);
            QuickSort(arr, start, partitionIndex-1);
            QuickSort(arr, partitionIndex+1, end);
        }
    }
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int N=scanner.nextInt();
        int[] arr = new int[N];
        for(int i=0;i<N;i++) arr[i]=scanner.nextInt();
        QuickSort(arr,0,N-1);
        for(int i=0;i<N;i++) System.out.print(arr[i]+" ");
        scanner.close();
    }
}