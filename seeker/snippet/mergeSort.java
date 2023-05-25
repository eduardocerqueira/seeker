//date: 2023-05-25T16:42:22Z
//url: https://api.github.com/gists/1a3af7dab6841737fb6067d4a9f64908
//owner: https://api.github.com/users/ashutoshdhande

import java.util.Arrays;

/**
 * MergeSort
 */
class MergeSort {
     public static void main(String[] args) {
          int[] arr = { 2, 4, 1, 6, 8, 5, 3, 7 };
          mergeSort(arr, arr.length);
          System.out.println(Arrays.toString(arr));
     }

     static void mergeSort(int[] arr, int n) {
          // int n = arr.length;
          if (n < 2)
               return;

          int mid = n / 2;
          int[] left = new int[mid];
          int[] right = new int[n - mid];

          for (int i = 0; i < mid; i++) {
               left[i] = arr[i];
          }
          for (int i = mid; i < n; i++) {
               right[i - mid] = arr[i];
          }

          mergeSort(left, mid);
          mergeSort(right, n - mid);
          merge(left, right, arr);
     }

     static void merge(int[] left, int[] right, int[] arr) {
          // int n = arr.length;
          int nl = left.length;
          int nr = right.length;
          int i = 0, j = 0, k = 0;
          while (i < nl && j < nr) {
               if (left[i] < right[j]) {
                    arr[k++] = left[i++];

               } else {
                    arr[k++] = right[j++];

               }
          }

          while (i < nl) {
               arr[k++] = left[i++];

          }

          while (j < nr) {
               arr[k++] = right[j++];

          }

     }

}