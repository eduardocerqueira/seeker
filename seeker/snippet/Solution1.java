//date: 2025-12-25T16:56:05Z
//url: https://api.github.com/gists/e4e9a9c62fccdc003baafd78330a1419
//owner: https://api.github.com/users/qornanali

package org.example.exercisev2.day11;

public class Solution1 {
    public double findMaxAverage(int[] nums, int k) {
        int left = 0;
        int right = 0;
        int n = nums.length;
        double sum = 0;

        while (right < n && right < k) {
            sum += nums[right];
            right++;
        }

        if (n < k) {
            return sum / n;
        }

        double ans = sum;

        while (right < n) {
            ans = Math.max(ans, sum);
            sum -= nums[left];
            sum += nums[right];
            left++;
            right++;
        }

        return Math.max(ans, sum) / k;
    }
}
