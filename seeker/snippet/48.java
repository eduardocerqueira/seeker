//date: 2022-03-09T17:07:38Z
//url: https://api.github.com/gists/6300b0ce6c16e8b23d98e69a1f82c8d9
//owner: https://api.github.com/users/TehleelMir

class Solution {
    public int removeCoveredIntervals(int[][] i)  {
        mergeSort(i, 0, i.length-1);
        int result = 0, right = 0;
        for(int[] x : i)
            if(x[1] > right) {
                result++;
                right = x[1];
            }
        return result;
    }

    void mergeSort(int[][] i, int left, int right) {
        if(left < right) {
            int mid = (left + right) / 2;
            mergeSort(i, left, mid);
            mergeSort(i, mid+1, right);
            merge(i, left, mid, right);
        }
    }

    void merge(int[][] brr, int left, int mid, int right) {
        int[][] arr = new int[brr.length+1][2]; 
        int i = left, j = mid+1, k = 0; 
        
        while(i <= mid && j <= right) {
            int one = brr[i][0];
            int two = brr[i][1];
            int oneB = brr[j][0];
            int twoB = brr[j][1];
            
            if(one < oneB) {
                arr[k++] = brr[i++];
            }
            else if(oneB < one) {
                arr[k++] = brr[j++];
            }
            else if(two > twoB) {
                arr[k++] = brr[i++];
            } 
            else if(twoB > two) {
                arr[k++] = brr[j++];
            }
        }
        
        while(i <= mid)
            arr[k++] = brr[i++];
        while(j <= right)
            arr[k++] = brr[j++];
        
        k=0;
        while(left <= right)
            brr[left++] = arr[k++];
    }
}