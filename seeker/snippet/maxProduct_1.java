//date: 2022-01-27T17:05:29Z
//url: https://api.github.com/gists/229abeff9f03c4292b0d13332eb6f6f2
//owner: https://api.github.com/users/akshay-ravindran-96

class Solution {
    public int maxProduct(int[] nums) {
        int max = nums[0];
        int min = nums[0];
        int res = nums[0];
        for( int i = 1; i< nums.length; i++)
            {
            int temp = min;
            min = Math.min(nums[i], Math.min(nums[i] * min, nums[i] * max));               
            max = Math.max( nums[i], Math.max(nums[i]  * max, nums[i] * temp));
            res = Math.max(max,res);
            
            }
        return res;
    }
}