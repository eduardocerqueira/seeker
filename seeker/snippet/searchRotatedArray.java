//date: 2024-12-27T16:57:44Z
//url: https://api.github.com/gists/4f6f8b9f7cc33a154a852ffe9a21adbd
//owner: https://api.github.com/users/solo-leveler

There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

//searchRotatedArray.java

class Solution {
    public int search(int[] nums, int target) {
        int start = 0;
        int end = nums.length - 1;
        int ans = 1;
        while (start <= end) {
            int mid = start + (end - start) / 2;
            if(target == nums[mid])
             return mid;
            // left side is sorted  
            if ( nums[start] <= nums[mid]){
                if (target >= nums[start] && target < nums[mid])
                    end = mid -1;
                else 
                    start = mid+1;
            }
            //right side is sorted 
            else {
                if(target > nums[mid] && target <= nums[end])
                    start = mid +1;
                else
                    end = mid -1;
            }
        }
        return -1;
    }
}