//date: 2021-12-29T17:19:02Z
//url: https://api.github.com/gists/3ee2a5010693b8c962a264bbc1c31767
//owner: https://api.github.com/users/ravichandrae

import java.util.Arrays;

public class LongestIncreasingSubsequence {
    public static void main(String[] args) {
        System.out.println(longestIncreasingSubsequence(null)); //Expected 0
        System.out.println(longestIncreasingSubsequence(new int[]{})); //Expected 0
        System.out.println(longestIncreasingSubsequence(new int[]{1})); //Expected 1
        System.out.println(longestIncreasingSubsequence(new int[]{1, 5, 10})); //Expected 3
        System.out.println(longestIncreasingSubsequence(new int[]{1, 5, 2, 7, 6, 8})); //Expected 4
        System.out.println(longestIncreasingSubsequence(new int[]{6, 1, 7, 3, 9, 4, 11, 13})); //Expected 5
        System.out.println(longestIncreasingSubsequence(new int[]{10, 22, 9, 33, 21, 50, 41, 60, 80})); //Expected 6
        System.out.println(longestIncreasingSubsequence(new int[]{5, 4, 1, 2, 3})); //Expected 3
        System.out.println(longestIncreasingSubsequence(new int[]{5, 2})); //Expected 1
        System.out.println(longestIncreasingSubsequence(new int[]{0, 1, 2, 0, 1, 3})); //Expected 4
        System.out.println(longestIncreasingSubsequence(new int[]{0, 1, 2, 3, 4, 5, 1, 3, 8})); //Expected 7
    }

    /*
    O(n^2) implementation - Dynamic programming
     */
    private static int longestIncreasingSubsequence(int[] arr) {
        int maxLength = 0;
        if (arr != null && arr.length > 0) {
            int[] longestSoFar = new int[arr.length];
            longestSoFar[0] = 1;
            for (int i = 1; i < arr.length; ++i) {
                for (int j = i - 1; j >= 0; j--) {
                    if (arr[j] < arr[i] && longestSoFar[j] > longestSoFar[i]) {
                        longestSoFar[i] = longestSoFar[j];
                    }
                }
                //Including the ith element
                longestSoFar[i]++;
            }
            maxLength = Arrays.stream(longestSoFar).max().getAsInt();
        }
        return maxLength;
    }
}
