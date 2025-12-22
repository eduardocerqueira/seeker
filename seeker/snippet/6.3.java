//date: 2025-12-22T17:05:05Z
//url: https://api.github.com/gists/4438e9bb5c60328faf15f53edc09fe64
//owner: https://api.github.com/users/JasonRon123

class ClimbingStairs {
    public int climbStairsTD(int n) {
        int[] memo = new int[n + 1];
        return climb(n, memo);
    }
    
    private int climb(int n, int[] memo) {
        if (n <= 2) return n;
        if (memo[n] != 0) return memo[n];
        memo[n] = climb(n - 1, memo) + climb(n - 2, memo);
        return memo[n];
    }
    
    public int climbStairsBU(int n) {
        if (n <= 2) return n;
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
    
    public int climbStairsBUOpt(int n) {
        if (n <= 2) return n;
        int first = 1;
        int second = 2;
        for (int i = 3; i <= n; i++) {
            int third = first + second;
            first = second;
            second = third;
        }
        return second;
    }
}