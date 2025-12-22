//date: 2025-12-22T17:06:19Z
//url: https://api.github.com/gists/7bc8224db299b696e9d0387d5f2b556d
//owner: https://api.github.com/users/JasonRon123

import java.util.Arrays;

class CoinChange {
    public int coinChangeTD(int[] coins, int amount) {
        Integer[] memo = new Integer[amount + 1];
        return dfs(coins, amount, memo);
    }
    
    private int dfs(int[] coins, int amount, Integer[] memo) {
        if (amount < 0) return -1;
        if (amount == 0) return 0;
        if (memo[amount] != null) return memo[amount];
        
        int min = Integer.MAX_VALUE;
        for (int coin : coins) {
            int res = dfs(coins, amount - coin, memo);
            if (res >= 0) {
                min = Math.min(min, res + 1);
            }
        }
        memo[amount] = min == Integer.MAX_VALUE ? -1 : min;
        return memo[amount];
    }
    
    public int coinChangeBU(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        
        for (int i = 1; i <= amount; i++) {
            for (int coin : coins) {
                if (coin <= i) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }
}