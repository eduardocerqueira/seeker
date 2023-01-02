//date: 2023-01-02T16:41:07Z
//url: https://api.github.com/gists/87b22eaf524a4c3fa00a248dd330bd0e
//owner: https://api.github.com/users/samarthsewlani

class Solution {
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        int xmax = nums[0];
        for(int i=0;i<n;i++){
            int xsum = 0;
            for(int j=i;j<n;j++){
                xsum += nums[j];
                xmax = Math.max(xmax,xsum);
            }
        }
        return xmax;
    }
}