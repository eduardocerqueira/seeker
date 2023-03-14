//date: 2023-03-14T16:56:21Z
//url: https://api.github.com/gists/7b837356634d34deb7e5aed079d9b73a
//owner: https://api.github.com/users/VallarasuS

class Solution {
    public int[] sortedSquares(int[] nums) {

        for(int i=0; i < nums.length; i++) {
            nums[i] *= nums[i];
        }
        
        this.quickSort(nums, 0, nums.length - 1);

        return nums;
    }
    
    public void quickSort(int[] nums, int left, int right) {
        
        if(left >= right) {
            return;
        }

        int pivot = this.partition(nums, left, right);

        this.quickSort(nums, left, pivot - 1);
        this.quickSort(nums, pivot + 1, right);
    }

    public int partition(int[] nums, int left, int right) {
        
        int swapPoint = left - 1;
        int pivot = nums[right];

        for(;left < right; left++) {

            if(nums[left] <= pivot) {
                swapPoint++;
                this.swap(nums, swapPoint, left);
            }
        }
        this.swap(nums, swapPoint + 1, right);
        return swapPoint + 1;
    }

    public void swap(int[] nums, int left, int right) {
        int item = nums[left];
        nums[left] = nums[right];
        nums[right] = item;
    }
}