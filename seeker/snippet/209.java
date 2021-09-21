//date: 2021-09-21T17:11:28Z
//url: https://api.github.com/gists/58e5a14d8e885d23a8f0276231cac1d3
//owner: https://api.github.com/users/kchiang6

class Solution {
    public int minSubArrayLen(int target, int[] nums) {
        int curShort = Integer.MAX_VALUE;
        
        int i = 0;
        int total = 0;
        int len = 0;
        
        for (int j = 0; j < nums.length; j++) {
            total += nums[j];
            len++;
            
            while (i <= j && target <= total) {
                curShort = Math.min(curShort, j-i+1);
                total -= nums[i];
                i++;
            }
        }
        
        return curShort == Integer.MAX_VALUE ? 0 : curShort;
    }
}