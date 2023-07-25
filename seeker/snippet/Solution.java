//date: 2023-07-25T17:00:39Z
//url: https://api.github.com/gists/885a85ad1fcc7abdd1272f6cc6caa944
//owner: https://api.github.com/users/diegoehg

/**
A solution to Third Maximum Problem stated here: https://leetcode.com/problems/third-maximum-number/
*/
class Solution {
    public int thirdMax(int[] nums) {
        Integer[] maximums = {null, null, null};

        for(int i:nums)
            compareNumberWithPosition(i, maximums, 0);

        if (maximums[2] != null)
            return maximums[2].intValue();

        return maximums[0].intValue();
    }

    private void compareNumberWithPosition(int i, Integer[] maximums, int index) {
        if (index >= 3)
            return;
        
        if (maximums[index] == null) {
            maximums[index] = i;
            return;
        }

        if (maximums[index] == i)
            return;

        int swap = i;
        if (maximums[index] < i) {
            swap = maximums[index].intValue();
            maximums[index] = i;
        }

        compareNumberWithPosition(swap, maximums, index + 1);
    }
}