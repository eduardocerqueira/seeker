//date: 2021-09-02T17:10:18Z
//url: https://api.github.com/gists/f53d4db563cab6e122b9f14ff011a9c6
//owner: https://api.github.com/users/jawadsab

class Solution {
    public int maxProfit(int[] prices) {
        int min = prices[0];
        int maxProfit = 0;
        for(int i=1; i<prices.length;i++) {
            
            if(prices[i] > min) {
                maxProfit = Math.max(maxProfit,prices[i]-min);
            }
            min = Math.min(min,prices[i]);
        }
        return maxProfit;
    }
}