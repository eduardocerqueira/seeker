//date: 2025-12-22T17:10:04Z
//url: https://api.github.com/gists/4af7d8d401e012fc734663ed0da574dc
//owner: https://api.github.com/users/JasonRon123

class HouseRobber {
    public int robTD(int[] nums) {
        Integer[] memo = new Integer[nums.length];
        return dfs(nums, 0, memo);
    }
    
    private int dfs(int[] nums, int index, Integer[] memo) {
        if (index >= nums.length) return 0;
        if (memo[index] != null) return memo[index];
        
        int rob = nums[index] + dfs(nums, index + 2, memo);
        int skip = dfs(nums, index + 1, memo);
        
        memo[index] = Math.max(rob, skip);
        return memo[index];
    }
    
    public int robBU(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        
        for (int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1], nums[i] + dp[i - 2]);
        }
        return dp[nums.length - 1];
    }
    
    public int robBUOpt(int[] nums) {
        int prev2 = 0;
        int prev1 = 0;
        
        for (int num : nums) {
            int current = Math.max(prev1, num + prev2);
            prev2 = prev1;
            prev1 = current;
        }
        return prev1;
    }
}