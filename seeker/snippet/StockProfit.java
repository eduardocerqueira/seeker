//date: 2023-10-02T17:09:23Z
//url: https://api.github.com/gists/1ce66f59751ff31cb9446c7b66958a94
//owner: https://api.github.com/users/oortegaa

public static final int LOWEST_PURCHASE = 2000;

public int maximumProfit(int[] stockPrices) {
    int profit = 0;
    int lowestPurchase = LOWEST_PURCHASE;
    int purchase = 0;
    for (int stock: stockPrices) {
        lowestPurchase = Math.min(lowestPurchase, stock);
        purchase = stock - lowestPurchase;
        profit = Math.max(profit, purchase);
    }
    return profit;
}