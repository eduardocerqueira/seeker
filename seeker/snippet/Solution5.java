//date: 2025-12-22T17:09:30Z
//url: https://api.github.com/gists/58d0294fb6ddfdc84a56722e39e12602
//owner: https://api.github.com/users/qornanali

package org.example.exercisev2.day6;

public class Solution5 {

    public int maxSubArray(int[] nums) {
        int currentSum = nums[0];
        int maxSum = nums[0];

        for (int i = 1; i < nums.length; i++) {
            currentSum = Math.max(nums[i], currentSum + nums[i]);
            maxSum = Math.max(maxSum, currentSum);
        }

        return maxSum;
    }
}
