//date: 2025-12-22T17:08:02Z
//url: https://api.github.com/gists/aa2eb0c4e89674d2a00b487fe1fa0091
//owner: https://api.github.com/users/JasonRon123

class Knapsack {
    public int knapsackTD(int[] weights, int[] values, int capacity) {
        int[][] memo = new int[weights.length][capacity + 1];
        for (int[] row : memo) java.util.Arrays.fill(row, -1);
        return dfs(weights, values, capacity, 0, memo);
    }
    
    private int dfs(int[] weights, int[] values, int capacity, int index, int[][] memo) {
        if (index == weights.length || capacity <= 0) return 0;
        if (memo[index][capacity] != -1) return memo[index][capacity];
        
        int take = 0;
        if (weights[index] <= capacity) {
            take = values[index] + dfs(weights, values, capacity - weights[index], index + 1, memo);
        }
        int skip = dfs(weights, values, capacity, index + 1, memo);
        
        memo[index][capacity] = Math.max(take, skip);
        return memo[index][capacity];
    }
    
    public int knapsackBU(int[] weights, int[] values, int capacity) {
        int n = weights.length;
        int[][] dp = new int[n + 1][capacity + 1];
        
        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= capacity; w++) {
                if (weights[i - 1] <= w) {
                    dp[i][w] = Math.max(
                        values[i - 1] + dp[i - 1][w - weights[i - 1]],
                        dp[i - 1][w]
                    );
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        return dp[n][capacity];
    }
    
    public int knapsackBUOpt(int[] weights, int[] values, int capacity) {
        int[] dp = new int[capacity + 1];
        
        for (int i = 0; i < weights.length; i++) {
            for (int w = capacity; w >= weights[i]; w--) {
                dp[w] = Math.max(dp[w], values[i] + dp[w - weights[i]]);
            }
        }
        return dp[capacity];
    }
}