//date: 2022-06-29T17:20:45Z
//url: https://api.github.com/gists/9841b37ab48267412ce683a6b5888f52
//owner: https://api.github.com/users/unitraveleryy

class Solution {
    public int maxProfit(int k, int[] prices) {
        int n = prices.length;
        if (n==0 || k==0) return 0;
        int[][][] dp = new int[n][2][k+1];
        
        for (int i = 0; i<k; i++) dp[n-1][1][i+1] = prices[n-1];
        
        for (int i = n-2; i >=0; i--) {
            for (int j = 1; j<=k; j++) {
                dp[i][0][j] = Math.max(dp[i+1][0][j],dp[i+1][1][j]-prices[i]);
                dp[i][1][j] = Math.max(dp[i+1][1][j],dp[i+1][0][j-1]+prices[i]);
            }
        }
        
        return dp[0][0][k];
    }
}

// Sampled optimal solution (2 ms)
class Solution {
    public int maxProfit(int k, int[] prices) {
        if(prices.length == 0) return 0;
        int[] buy = new int[k + 1];
        int[] sell = new int [k + 1];
        Arrays.fill(buy, Integer.MIN_VALUE / 2);
        Arrays.fill(sell, Integer.MIN_VALUE / 2);
        buy[0] = - prices[0];
        sell[0] = 0;
        for(int i = 1; i < prices.length; i++){
            buy[0] = Math.max(buy[0], sell[0]-prices[i]);
            for(int j = 1; j <= k; j++){
                buy[j] = Math.max(buy[j], sell[j] - prices[i]);
                sell[j] = Math.max(sell[j], buy[j - 1] + prices[i]);
            }
        }
        int res = 0;
        for(int j = 1; j <= k; j++){
            res = Math.max(sell[j], res);
        }
        return res;
    }
}