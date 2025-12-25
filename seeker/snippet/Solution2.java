//date: 2025-12-25T17:03:12Z
//url: https://api.github.com/gists/20579089c98858adc17f24fb5e363bc1
//owner: https://api.github.com/users/qornanali

package org.example.exercisev2.day11;

public class Solution2 {
    public void moveZeroes(int[] nums) {
        int nonZeroPointer = 0;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                int tmp = nums[i];
                nums[i] = nums[nonZeroPointer];
                nums[nonZeroPointer] = tmp;
                nonZeroPointer++;
            }
        }
    }
}
