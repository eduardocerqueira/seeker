//date: 2025-12-22T16:54:53Z
//url: https://api.github.com/gists/190b835b59a1733cc7e328a1c5a03896
//owner: https://api.github.com/users/JasonRon123

class Solution121 {
    public int maxProfit(int[] prices) {
        int minPrice = Integer.MAX_VALUE;
        int maxProfit = 0;
        for (int price : prices) {
            if (price < minPrice) {
                minPrice = price;
            } else if (price - minPrice > maxProfit) {
                maxProfit = price - minPrice;
            }
        }
        return maxProfit;
    }
}