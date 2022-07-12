//date: 2022-07-12T17:08:19Z
//url: https://api.github.com/gists/71515514d3b50e3a05a6facab561426e
//owner: https://api.github.com/users/unitraveleryy

class Solution {
    public int integerBreak(int n) {
        int[] dp = new int[n+1];
        if (n == 2) return 1;
        dp[1] = 1;
        dp[2] = 1;
        for (int i = 3; i<=n; i++) {
            int candid1, candid2;
            if (i-2<=3) candid1 = 2*(i-2);
            else candid1 = 2*dp[i-2];
            if (i-3<=3) candid2 = 3*(i-3);
            else candid2 = 3*dp[i-3];
            dp[i] = Math.max(candid1,candid2);
        }
        // why considering only -2 and -3?
        // because, say, e.g., -4, 4 can be decomposed as 2 + 2 = 4, the product is still 4
        // and say 5 = 2 + 3, 2 x 3 > 5.
        
        return dp[n];
    }
}