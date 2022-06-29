//date: 2022-06-29T17:18:36Z
//url: https://api.github.com/gists/782e8a68e2d2963469144c8134d28a72
//owner: https://api.github.com/users/unitraveleryy

// class Solution {
//     public int maxProfit(int[] prices) {
//         int n = prices.length;
//         int[][][] dp = new int[n][2][2];
//         dp[n-1][1][1] = prices[n-1];
//         // if optimize
//         // dp[i+1][1 for have stock][left selling time]
//         for (int i = n-2; i>=0; i--) {
//             dp[i][1][1] = Math.max(dp[i+1][0][0]+prices[i],dp[i+1][1][1]);
//             dp[i][0][1] = Math.max(dp[i+1][1][1]-prices[i],dp[i+1][0][1]);
//         }
//         return dp[0][0][1];
//     }
// }

// class Solution {
//     public int maxProfit(int[] prices) {
//         int n = prices.length;
//         int[][] dp = new int[n][2];
//         dp[n-1][1] = prices[n-1];
//         // if optimize
//         // dp[i+1][1 for have stock][left selling time]
//         for (int i = n-2; i>=0; i--) {
//             dp[i][1] = Math.max(prices[i],dp[i+1][1]);
//             dp[i][0] = Math.max(dp[i+1][1]-prices[i],dp[i+1][0]);
//         }
//         return dp[0][0];
//     }
// }

class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int lowestSoFar = prices[0];
        int profit = 0;
        
        for (int i = 1; i<n; i++) {
            if (prices[i] < lowestSoFar) lowestSoFar = prices[i];
            else profit = Math.max(profit, prices[i]-lowestSoFar);
        }
        return profit;
    }
}

