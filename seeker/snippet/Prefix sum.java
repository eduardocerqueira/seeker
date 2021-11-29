//date: 2021-11-29T17:16:11Z
//url: https://api.github.com/gists/1aa0951982c5dad916b66a578624ca5d
//owner: https://api.github.com/users/Vik1ang

class NumArray {

    private int[] preSum;

    public NumArray(int[] nums) {
        // preSum[0] = 0, 便于计算累加积
        preSum = new int[nums.length + 1];
        // 计算 nums 的累加积
        for (int i = 1; i < preSum.length; i++) {
            preSum[i] = preSum[i - 1] + nums[i - 1];
        }
    }
    
    public int sumRange(int left, int right) {
        return preSum[right + 1] - preSum[left];
    }
    
}