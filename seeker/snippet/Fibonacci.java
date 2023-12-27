//date: 2023-12-27T17:06:47Z
//url: https://api.github.com/gists/ff018d7ebf03144bb377c3b91fafb4c8
//owner: https://api.github.com/users/nitverse

import java.util.Arrays;

public class Fibonacci {

    // Using Memoization
    private static int memoize(int n, int[] dp) {
        if (n <= 1) return n;
        if (dp[n] != -1) return dp[n];
        return dp[n] = memoize(n - 1, dp) + memoize(n - 2, dp);
    }

    // Using Tabulation
    private static int tabulate(int n, int[] dp) {
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    // Space-Optimized method (Optimal)
    private static int optimize(int n) {
        int prev2 = 0, prev = 1;
        int currentIndex = 0;
        for (int i = 2; i <= n; i++) {
            currentIndex = prev + prev2;
            prev2 = prev;
            prev = currentIndex;
        }
        return prev;
    }

    public static void main(String[] args) {
        int n = 15;
        int[] dp = new int[n + 1];
        Arrays.fill(dp, -1);

        // Memoization
        // System.out.println(memoize(n, dp));

        // Tabulation
        // Convert recursive solution to tabulation
        System.out.println(tabulate(n, dp));

        // Space-Optimized
        System.out.println(optimize(n));
    }
}
