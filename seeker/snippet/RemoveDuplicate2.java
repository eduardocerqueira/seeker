//date: 2025-04-03T17:02:09Z
//url: https://api.github.com/gists/7d6f86b1237564d560789a84a5432449
//owner: https://api.github.com/users/qren0neu

class Solution {
    public int removeDuplicates(int[] nums) {
        int duplicate = nums[0];
        int duplicateCount = 1;
        int maxAllowedDuplicateCount = 2;
        int slow = 1;
        for (int fast = 1; fast < nums.length; fast++) {
            if (nums[fast] != duplicate) {
                nums[slow] = nums[fast];
                slow++;
                // reset duplicate count
                duplicateCount = 1;
            } else if (duplicateCount < maxAllowedDuplicateCount) {
                // duplicate but not exceed
                nums[slow] = nums[fast];
                slow++;
                duplicateCount++;
            }
            duplicate = nums[fast];
        }
        return slow;
    }
}