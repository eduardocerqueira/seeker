//date: 2022-03-09T16:59:16Z
//url: https://api.github.com/gists/0d51c0daf8eb138c5e783f8f7117e2ce
//owner: https://api.github.com/users/TehleelMir

import java.util.Arrays;

public class QuickSort {
    public static void main(String[] args) {
        int[] arr = {3,5,9,99,0,1,5,2};
        quickSort(arr, 0, arr.length-1);
        System.out.print(Arrays.toString(arr));
    }

    public static void quickSort(int[] arr, int low, int high) {
        if(low > high) return;
        int pivot = partition(arr, low, high);
        quickSort(arr, low, pivot-1);
        quickSort(arr, pivot+1, high);
    }

    public static int partition(int[] arr, int low, int high) {
        //low will be our pivot
        int nextPivot = low+1;
        for(int i=nextPivot; i<=high; i++)
            if(arr[i] < arr[low])
                swap(arr, i, nextPivot++);
        swap(arr, low, nextPivot-1);
        return nextPivot-1;
    }

    public static void swap(int[] arr, int _this, int _withThis) {
        int temp = arr[_this];
        arr[_this] = arr[_withThis];
        arr[_withThis] = temp;
    }
}