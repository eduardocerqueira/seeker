//date: 2022-01-27T17:02:32Z
//url: https://api.github.com/gists/96f5fe9b198657085682169408d5becd
//owner: https://api.github.com/users/akshay-ravindran-96

class Solution {
    public int[] productExceptSelf(int[] nums) {
        
        int total = 1;
        int second = 1;
        int z=0;
        for(int i : nums)
        {
            if(i!=0)
                second *=i;
            else
                z++;
            
            total *=i;
        }
        
        int ans[] = new int[nums.length];
        
        if(z>1)
            return ans;
        
        for(int i=0; i< nums.length; i++)
        {
            if(nums[i] == 0)
                ans[i] = second;
            else
                ans[i] = total/nums[i];
        }
        
        return ans;
        
    }
}