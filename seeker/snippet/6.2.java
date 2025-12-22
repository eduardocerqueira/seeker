//date: 2025-12-22T17:02:57Z
//url: https://api.github.com/gists/032f22b0bdbc809a61b906528d9dff89
//owner: https://api.github.com/users/JasonRon123

import java.util.Arrays;

public class DynamicProgrammingAnalysis {
    
    public static long fibonacciTopDown(int n) {
        long[] memo = new long[n + 1];
        Arrays.fill(memo, -1);
        return fibHelper(n, memo);
    }
    
    private static long fibHelper(int n, long[] memo) {
        if (n <= 1) return n;
        if (memo[n] != -1) return memo[n];
        memo[n] = fibHelper(n - 1, memo) + fibHelper(n - 2, memo);
        return memo[n];
    }
    
    public static long fibonacciBottomUp(int n) {
        if (n <= 1) return n;
        long prev2 = 0;
        long prev1 = 1;
        long current = 0;
        for (int i = 2; i <= n; i++) {
            current = prev1 + prev2;
            prev2 = prev1;
            prev1 = current;
        }
        return current;
    }
    
    public static int coinChangeTopDown(int[] coins, int amount) {
        Integer[] memo = new Integer[amount + 1];
        return coinHelper(coins, amount, memo);
    }
    
    private static int coinHelper(int[] coins, int amount, Integer[] memo) {
        if (amount < 0) return -1;
        if (amount == 0) return 0;
        if (memo[amount] != null) return memo[amount];
        
        int minCoins = Integer.MAX_VALUE;
        for (int coin : coins) {
            int res = coinHelper(coins, amount - coin, memo);
            if (res >= 0 && res < minCoins) {
                minCoins = res + 1;
            }
        }
        memo[amount] = (minCoins == Integer.MAX_VALUE) ? -1 : minCoins;
        return memo[amount];
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
    
    public static int coinChangeBottomUpOptimized(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        
        for (int coin : coins) {
            for (int i = coin; i <= amount; i++) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }
    
    public static int lcsTopDown(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        int[][] memo = new int[m][n];
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
        int m = text1.length();
        int n = text2.length();
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
    
    public static int lcsBottomUpOptimized(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        if (m < n) {
            return lcsBottomUpOptimized(text2, text1);
        }
        
        int[] prev = new int[n + 1];
        int[] curr = new int[n + 1];
        
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    curr[j] = prev[j - 1] + 1;
                } else {
                    curr[j] = Math.max(prev[j], curr[j - 1]);
                }
            }
            int[] temp = prev;
            prev = curr;
            curr = temp;
            Arrays.fill(curr, 0);
        }
        return prev[n];
    }
    
    public static void main(String[] args) {
        System.out.println("=== СРАВНЕНИЕ ПОДХОДОВ ===\n");
        
        System.out.println("1. ЧИСЛА ФИБОНАЧЧИ (n=40):");
        long startTime = System.nanoTime();
        long fibTD = fibonacciTopDown(40);
        long tdTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        long fibBU = fibonacciBottomUp(40);
        long buTime = System.nanoTime() - startTime;
        
        System.out.printf("Top-down: %d, Time: %d ns%n", fibTD, tdTime);
        System.out.printf("Bottom-up: %d, Time: %d ns%n", fibBU, buTime);
        System.out.printf("Восходящий быстрее в %.2f раз%n", (double)tdTime/buTime);
        
        System.out.println("\n2. РАЗМЕН МОНЕТ (coins=[1,2,5], amount=1000):");
        int[] coins = {1, 2, 5};
        startTime = System.nanoTime();
        int coinTD = coinChangeTopDown(coins, 1000);
        tdTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        int coinBU = coinChangeBottomUp(coins, 1000);
        buTime = System.nanoTime() - startTime;
        
        System.out.printf("Top-down: %d, Time: %d ns%n", coinTD, tdTime);
        System.out.printf("Bottom-up: %d, Time: %d ns%n", coinBU, buTime);
        System.out.printf("Восходящий быстрее в %.2f раз%n", (double)tdTime/buTime);
        
        System.out.println("\n3. LCS (две строки по 1000 символов):");
        StringBuilder sb1 = new StringBuilder();
        StringBuilder sb2 = new StringBuilder();
        for (int i = 0; i < 1000; i++) {
            sb1.append((char)('A' + (i % 26)));
            sb2.append((char)('A' + ((i * 7) % 26)));
        }
        String text1 = sb1.toString();
        String text2 = sb2.toString();
        
        startTime = System.nanoTime();
        int lcsTD = lcsTopDown(text1, text2);
        tdTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        int lcsBU = lcsBottomUp(text1, text2);
        buTime = System.nanoTime() - startTime;
        
        System.out.printf("Top-down: %d, Time: %d ns%n", lcsTD, tdTime);
        System.out.printf("Bottom-up: %d, Time: %d ns%n", lcsBU, buTime);
        System.out.printf("Восходящий быстрее в %.2f раз%n", (double)tdTime/buTime);
        
        System.out.println("\n=== АНАЛИЗ СЛОЖНОСТИ ===");
        System.out.println("1. Фибоначчи:");
        System.out.println("   Top-down: O(n) время, O(n) память (стек вызовов + мемоизация)");
        System.out.println("   Bottom-up: O(n) время, O(1) память (с оптимизацией)");
        
        System.out.println("\n2. Размен монет:");
        System.out.println("   Top-down: O(amount * coins) время, O(amount) память");
        System.out.println("   Bottom-up: O(amount * coins) время, O(amount) память");
        
        System.out.println("\n3. LCS:");
        System.out.println("   Top-down: O(m*n) время, O(m*n) память");
        System.out.println("   Bottom-up: O(m*n) время, O(m*n) память");
        System.out.println("   Оптимизированный: O(m*n) время, O(min(m,n)) память");
        
        System.out.println("\n=== ОПТИМИЗАЦИЯ ПРОСТРАНСТВА (LCS) ===");
        System.out.println("Исходная версия использует O(m*n) памяти.");
        System.out.println("Оптимизированная версия использует два массива размером O(n):");
        System.out.println("- Храним только предыдущую и текущую строки DP таблицы");
        System.out.println("- Этого достаточно, так как для вычисления dp[i][j] нужны:");
        System.out.println("  dp[i-1][j], dp[i][j-1], dp[i-1][j-1]");
        
        System.out.println("\nПроверка оптимизированной LCS:");
        startTime = System.nanoTime();
        int lcsBUOpt = lcsBottomUpOptimized(text1, text2);
        long optTime = System.nanoTime() - startTime;
        System.out.printf("Обычный bottom-up: %d ns%n", buTime);
        System.out.printf("Оптимизированный: %d ns, результат: %d%n", optTime, lcsBUOpt);
        System.out.println("Результаты совпадают: " + (lcsBU == lcsBUOpt));
    }
}