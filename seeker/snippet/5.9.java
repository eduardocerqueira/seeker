//date: 2025-12-22T16:54:11Z
//url: https://api.github.com/gists/d27c436d374017707ec415138c3c7ab5
//owner: https://api.github.com/users/JasonRon123

class Solution55 {
    public boolean canJump(int[] nums) {
        int maxReach = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > maxReach) return false;
            maxReach = Math.max(maxReach, i + nums[i]);
            if (maxReach >= nums.length - 1) return true;
        }
        return false;
    }
}