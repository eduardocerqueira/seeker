//date: 2023-12-19T16:37:52Z
//url: https://api.github.com/gists/788c3049917e523b282a8b20be4c331a
//owner: https://api.github.com/users/quiode

// ADDITIONAL IMPORTS ARE NOT ALLOWED

class Main {
  public static void main(String[] args) {
    // Uncomment the following two lines if you want to read from a file
    In.open("public/example.in");
    Out.compareTo("public/example.out");

    int t = In.readInt(); // number of tests
    for (int test = 0; test < t; test++) {
      int n = In.readInt(); // size of A
      int[] A = new int[n];
      for (int i = 0; i < n; i++) {
        A[i] = In.readInt();
      }
      Out.println(getMaximumScore(n, A));
    }
    
    // Uncomment this line if you want to read from a file
    In.close();
  }
  
  public static int getMaximumScore(int n, int[] A) {
    int[][] dp = new int[n][n];
    
    for (int i = n-1; i >= 0; i--) {
      for (int j = 0; j < n; j++) {
        // base case
        if (i >= j) {
          dp[i][j] = 0;
        } else {
          // recursion
          int result = Integer.MIN_VALUE;
          
          if (i < n-1 && j > 0) {
            result = Math.max(result, Math.abs(A[i] - A[j]) + dp[i+1][j-1]);
          }
          
          if (i < n-1) {
            result = Math.max(result, dp[i+1][j]);
          }
          
          if (j > 0) {
            result = Math.max(result, dp[i][j-1]);
          }
         
          dp[i][j] = result;
        }
      }
    }
    
    return dp[0][n-1];
  }
}