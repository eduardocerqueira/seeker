//date: 2022-01-27T17:06:33Z
//url: https://api.github.com/gists/43bbe2b0c72ec782f03edd559ff1ed34
//owner: https://api.github.com/users/akshay-ravindran-96

class Solution {
    public int findMin(int[] nums) {
        int left = 0;
        int right = nums.length -1;
        
        while(left < right)
            {
            if(nums[left] < nums[right])
                break;
            int mid = left + (right - left)/2;
            
            if(nums[mid] > nums[right])
                left = mid+1;
            
            else
                right = mid;
            }
        return nums[left];
    }
}