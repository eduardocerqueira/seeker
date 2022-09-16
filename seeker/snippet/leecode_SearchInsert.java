//date: 2022-09-16T21:10:50Z
//url: https://api.github.com/gists/2bc6a1dfb5bbcf9357c1d7991dd0383f
//owner: https://api.github.com/users/ChinaYC

/**
 * 数在数组中顺序插入
 */
public class SearchInsert {

    public int searchInsert1(int[] nums, int target) {
        //暴力遍历
        for(int i = 0; i < nums.length; i++){
            if(target <= nums[i]){
                return i;
            }
        }
        return nums.length;
    }

    public int searchInsert(int[] nums, int target) {
        // 二分法
        int left = 0, right = nums.length - 1;
        while(left <= right){
            // 防止 left+right 整型溢出
            int mid = left + (right - left) / 2;
            if(nums[mid] == target){
                return mid;
            }else if(nums[mid] < target){
                left = mid + 1;
            }else if(nums[mid] > target){
                right = mid - 1;
            }
        }
        return left;
    }

}