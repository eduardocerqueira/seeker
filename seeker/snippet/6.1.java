//date: 2025-12-22T17:00:05Z
//url: https://api.github.com/gists/74a8f9b90c8d0f444589539ff5874b5c
//owner: https://api.github.com/users/JasonRon123

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class DynamicProgrammingTasks {
    
    // 1. Числа Фибоначчи
    public static long fibonacciTopDown(int n) {
        Map<Integer, Long> memo = new HashMap<>();
        return fibHelper(n, memo);
    }
    
    private static long fibHelper(int n, Map<Integer, Long> memo) {
        if (n <= 1) return n;
        if (memo.containsKey(n)) return memo.get(n);
        long result = fibHelper(n - 1, memo) + fibHelper(n - 2, memo);
        memo.put(n, result);
        return result;
    }
    
    public static long fibonacciBottomUp(int n) {
        if (n <= 1) return n;
        long[] dp = new long[n + 1];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
    
    // 2. Размен монет
    public static int coinChangeTopDown(int[] coins, int amount) {
        Map<Integer, Integer> memo = new HashMap<>();
        int result = coinHelper(coins, amount, memo);
        return result == Integer.MAX_VALUE ? -1 : result;
    }
    
    private static int coinHelper(int[] coins, int amount, Map<Integer, Integer> memo) {
        if (amount == 0) return 0;
        if (amount < 0) return Integer.MAX_VALUE;
        if (memo.containsKey(amount)) return memo.get(amount);
        
        int minCoins = Integer.MAX_VALUE;
        for (int coin : coins) {
            int subResult = coinHelper(coins, amount - coin, memo);
            if (subResult != Integer.MAX_VALUE) {
                minCoins = Math.min(minCoins, subResult + 1);
            }
        }
        memo.put(amount, minCoins);
        return minCoins;
    }
    
    public static int coinChangeBottomUp(int[] coins, int amount) {
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
    
    // 3. Наибольшая общая подпоследовательность
    public static int lcsTopDown(String text1, String text2) {
        int[][] memo = new int[text1.length()][text2.length()];
        for (int[] row : memo) Arrays.fill(row, -1);
        return lcsHelper(text1, text2, 0, 0, memo);
    }
    
    private static int lcsHelper(String text1, String text2, int i, int j, int[][] memo) {
        if (i == text1.length() || j == text2.length()) return 0;
        if (memo[i][j] != -1) return memo[i][j];
        
        if (text1.charAt(i) == text2.charAt(j)) {
            memo[i][j] = 1 + lcsHelper(text1, text2, i + 1, j + 1, memo);
        } else {
            memo[i][j] = Math.max(
                lcsHelper(text1, text2, i + 1, j, memo),
                lcsHelper(text1, text2, i, j + 1, memo)
            );
        }
        return memo[i][j];
    }
    
    public static int lcsBottomUp(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m + 1][n + 1];
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        return dp[m][n];
    }
    
    public static void main(String[] args) {
        // Тестирование
        System.out.println("Fibonacci (n=35):");
        long startTime = System.nanoTime();
        long fibTD = fibonacciTopDown(35);
        long tdTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        long fibBU = fibonacciBottomUp(35);
        long buTime = System.nanoTime() - startTime;
        
        System.out.println("Top-down: " + fibTD + ", Time: " + tdTime + " ns");
        System.out.println("Bottom-up: " + fibBU + ", Time: " + buTime + " ns");
        
        System.out.println("\nCoin Change:");
        int[] coins = {1, 2, 5};
        int amount = 11;
        int coinTD = coinChangeTopDown(coins, amount);
        int coinBU = coinChangeBottomUp(coins, amount);
        System.out.println("Top-down: " + coinTD);
        System.out.println("Bottom-up: " + coinBU);
        
        System.out.println("\nLCS:");
        String text1 = "AGGTAB";
        String text2 = "GXTXAYB";
        int lcsTD = lcsTopDown(text1, text2);
        int lcsBU = lcsBottomUp(text1, text2);
        System.out.println("Top-down: " + lcsTD);
        System.out.println("Bottom-up: " + lcsBU);
    }
}