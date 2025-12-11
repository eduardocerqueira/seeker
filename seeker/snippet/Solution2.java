//date: 2025-12-11T17:07:46Z
//url: https://api.github.com/gists/b2db229a2cf88bc870a0bf241b3cb051
//owner: https://api.github.com/users/qornanali

package org.example.exercisev2.day4;

public class Solution2 {
    public int jump(int[] nums) {
        int n = nums.length;
        if (n == 1) {
            return 0;
        }

        int jumps = 0;
        int currentEnd = 0;
        int furthest = 0;

        for (int i = 0; i < n - 1; i++) {

            furthest = Math.max(furthest, i + nums[i]);

            if (i == currentEnd) {
                jumps++;
                currentEnd = furthest;
                if (currentEnd >= n - 1) {
                    break;
                }
            }
        }

        return jumps;
    }
}
