//date: 2025-12-25T17:09:40Z
//url: https://api.github.com/gists/3f9928d482b88983bb83313120f5102a
//owner: https://api.github.com/users/samarthsewlani

class Solution {
    public int pivotIndex(int[] arr) {
        int rsum = 0, n = arr.length, lsum = 0;
        for(int i=0;i<n;i++)    rsum += arr[i];
        for(int i=0;i<n;i++){
            rsum -= arr[i];
            if(lsum == rsum){
                return i;
            }
            lsum += arr[i];
        }
        return -1;
    }
}