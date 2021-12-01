//date: 2021-12-01T17:15:44Z
//url: https://api.github.com/gists/656d0061618d42eb5b1c486d88617d3a
//owner: https://api.github.com/users/mirzacc

class Solution {
    public void swap(int[] ar, int i , int j){
        int temp = ar[i];
        ar[i] = ar[j];
        ar[j] = temp;
    }
    public void nextPermutation(int[] nums) {
        int br = -1;
        int ma = -1;
        for(int i = nums.length-2; i >= 0; i--){
            if(nums[i] < nums[i+1]){
                br = i;
                break;
            }
        }
        if(br != -1){
            for(int i = nums.length -1; i >= 0; i--){
                if(nums[i] > nums[br]){
                    ma = i;
                    break;
                }
            }
            if(ma != -1){
                swap(nums,br,ma);
                int j = nums.length - 1;
                int i = br+1;
                while(i < j){
                    swap(nums,i,j);
                    i++;
                    j--;
                }
            }
            else
                br = -1;
        }
        if(br == -1){
            int j = nums.length - 1;
            int i = 0;
            while(i < j){
                swap(nums,i,j);
                i++;
                j--;
            }
        }
    }
}