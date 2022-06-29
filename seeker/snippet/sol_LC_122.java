//date: 2022-06-29T17:19:10Z
//url: https://api.github.com/gists/fb4542bc1655db704afeecdfbab940c4
//owner: https://api.github.com/users/unitraveleryy

class Solution {
    public int maxProfit(int[] prices) {
        int n = prices.length;
        int[][] dp = new int[n][2];
        
        dp[n-1][0] = 0;
        dp[n-1][1] = prices[n-1];
        
        for (int i = n-2; i>=0; i--) {
            dp[i][0] = Math.max(dp[i+1][1]-prices[i], dp[i+1][0]);
            dp[i][1] = Math.max(dp[i+1][0]+prices[i], dp[i+1][1]);
        }
        
        return dp[0][0];
    }
}