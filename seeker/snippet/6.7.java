//date: 2025-12-22T17:09:14Z
//url: https://api.github.com/gists/727dfbdb5eb95f1d96e5087647634e28
//owner: https://api.github.com/users/JasonRon123

class LongestCommonSubsequence {
    public int longestCommonSubsequenceTD(String text1, String text2) {
        int[][] memo = new int[text1.length()][text2.length()];
        for (int[] row : memo) java.util.Arrays.fill(row, -1);
        return dfs(text1, text2, 0, 0, memo);
    }
    
    private int dfs(String text1, String text2, int i, int j, int[][] memo) {
        if (i == text1.length() || j == text2.length()) return 0;
        if (memo[i][j] != -1) return memo[i][j];
        
        if (text1.charAt(i) == text2.charAt(j)) {
            memo[i][j] = 1 + dfs(text1, text2, i + 1, j + 1, memo);
        } else {
            memo[i][j] = Math.max(
                dfs(text1, text2, i + 1, j, memo),
                dfs(text1, text2, i, j + 1, memo)
            );
        }
        return memo[i][j];
    }
    
    public int longestCommonSubsequenceBU(String text1, String text2) {
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
}