//date: 2025-09-01T17:04:55Z
//url: https://api.github.com/gists/311b51d168aa2481c27254a558980229
//owner: https://api.github.com/users/kousthubha-sky

//1752. Check if Array Is Sorted and Rotated
class Solution {
    public boolean check(int[] nums) {
        int n = nums.length;
        int count = 0;
        for(int i=0;i<n-1;i++){
            if(nums[i]>nums[i+1]){
                count++;
                if(count>1) return false;
            }
        }
        if(count==1&&nums[n-1]>nums[0]){
            return false;
        }
        return true;
    }
}