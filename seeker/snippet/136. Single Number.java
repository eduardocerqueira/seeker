//date: 2022-10-20T17:31:23Z
//url: https://api.github.com/gists/68099fc5046ea8edac1cbaf39b94c9b5
//owner: https://api.github.com/users/cmichaelsd

class Solution {
    public int singleNumber(int[] nums) {
        var result = 0;
        for (int n : nums) {
            result ^= n;
        }
        return result;
    }
}