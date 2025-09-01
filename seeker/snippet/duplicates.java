//date: 2025-09-01T17:04:55Z
//url: https://api.github.com/gists/311b51d168aa2481c27254a558980229
//owner: https://api.github.com/users/kousthubha-sky

//26. Remove Duplicates from Sorted Array
class Solution {
    public int removeDuplicates(int[] nums) {
        int j = 0;
        for(int i = 1;i  < nums.length; i++){
            if(nums[j]!=nums[i]){
                nums[j+1]= nums[i];
                j++;
            }
        }
        return j+1;
    }
}