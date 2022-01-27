//date: 2022-01-27T17:08:52Z
//url: https://api.github.com/gists/0e211e50db23eb9e1859a2fa13f60c65
//owner: https://api.github.com/users/anil477

class Solution {
    // https://leetcode.com/problems/find-pivot-index/
    // Time Complexity: O(N)
    // Space Complexity: O(1)
    public int pivotIndex(int[] nums) {
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            sum = sum + nums[i];
        }
        int sumTillHere = 0;
        for (int i = 0; i < nums.length; i++) {
            if ((sum - nums[i] - sumTillHere) == sumTillHere) {
                return i;    
            }
            sumTillHere += nums[i];   
        }
        return -1;
    }
}