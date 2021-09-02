//date: 2021-09-02T17:16:31Z
//url: https://api.github.com/gists/b8995c02661db586891c9a8423df1f6b
//owner: https://api.github.com/users/jawadsab

class Solution {
    public int maxProfit(int[] prices) {
        int maxProfit = 0;
        for(int i=1; i<prices.length; i++) {
            if(prices[i] > prices[i-1]) {
                maxProfit += prices[i] - prices[i-1];
            }
        }
        return maxProfit;
    }
}