//date: 2025-09-01T17:04:55Z
//url: https://api.github.com/gists/311b51d168aa2481c27254a558980229
//owner: https://api.github.com/users/kousthubha-sky

//136. Single Number
class Solution {
    public int singleNumber(int[] nums) {
        int single = 0;
        for (int num : nums) {
            single ^= num; // XOR accumulates the single
        }
        return single;
    }
}
