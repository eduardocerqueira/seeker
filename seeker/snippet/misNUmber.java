//date: 2025-09-01T17:04:55Z
//url: https://api.github.com/gists/311b51d168aa2481c27254a558980229
//owner: https://api.github.com/users/kousthubha-sky

//268. Missing Number
class Solution {
    public int missingNumber(int[] nums) {
        int n = nums.length;
        int s = 0;
        int sum = n * (n + 1) / 2;
        for (int i = 0; i < nums.length; i++) {
            s += nums[i];
        }
        return sum - s;
    }
}