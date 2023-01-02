//date: 2023-01-02T16:40:10Z
//url: https://api.github.com/gists/8cabe1b65c807785316f20205cc928e0
//owner: https://api.github.com/users/samarthsewlani

class Solution {
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        int xsum = nums[0];
        int xmax = nums[0];
        for(int i=1;i<n;i++){
            int dec1 = xsum + nums[i];
            int dec2 = nums[i];
            xsum = Math.max(dec1,dec2);
            xmax = Math.max(xmax,xsum);
        }
        return xmax;
    }
}