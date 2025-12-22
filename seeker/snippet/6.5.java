//date: 2025-12-22T17:07:11Z
//url: https://api.github.com/gists/8db77ec6d7425eec0c05fe4dae6a5808
//owner: https://api.github.com/users/JasonRon123

import java.util.Arrays;

class LongestIncreasingSubsequence {
    public int lengthOfLISTD(int[] nums) {
        int[][] memo = new int[nums.length][nums.length + 1];
        for (int[] row : memo) Arrays.fill(row, -1);
        return dfs(nums, -1, 0, memo);
    }
    
    private int dfs(int[] nums, int prevIndex, int currIndex, int[][] memo) {
        if (currIndex == nums.length) return 0;
        if (memo[currIndex][prevIndex + 1] != -1) return memo[currIndex][prevIndex + 1];
        
        int take = 0;
        if (prevIndex == -1 || nums[currIndex] > nums[prevIndex]) {
            take = 1 + dfs(nums, currIndex, currIndex + 1, memo);
        }
        int skip = dfs(nums, prevIndex, currIndex + 1, memo);
        
        memo[currIndex][prevIndex + 1] = Math.max(take, skip);
        return memo[currIndex][prevIndex + 1];
    }
    
    public int lengthOfLISBU(int[] nums) {
        if (nums.length == 0) return 0;
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        
        int maxLen = 1;
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            maxLen = Math.max(maxLen, dp[i]);
        }
        return maxLen;
    }
    
    public int lengthOfLISOptimal(int[] nums) {
        int[] tails = new int[nums.length];
        int size = 0;
        
        for (int num : nums) {
            int left = 0, right = size;
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (tails[mid] < num) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            tails[left] = num;
            if (left == size) size++;
        }
        return size;
    }
}