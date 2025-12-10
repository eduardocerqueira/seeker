//date: 2025-12-10T17:05:55Z
//url: https://api.github.com/gists/0621d7cc16ec7691c62d03ba389503b1
//owner: https://api.github.com/users/qornanali

package org.example.exercisev2.day3;

public class Solution2 {
    public int maxProfit(int[] prices) {
        int bestProfit = 0;
        int n = prices.length;
        int right = 1;
        int left = 0;

        while (right < n) {
            if (prices[right] < prices[left]) {
                left = right;
            } else {
                bestProfit = Math.max(bestProfit, prices[right] - prices[left]);
            }
            right++;
        }

        return bestProfit;
    }
}
