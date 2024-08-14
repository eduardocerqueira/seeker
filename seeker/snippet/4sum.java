//date: 2024-08-14T18:45:09Z
//url: https://api.github.com/gists/32edde2a2bd89a085281486312a6892d
//owner: https://api.github.com/users/kymotz

class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<>();
        for(int k = 0; k < nums.length; k++) {
            if (nums[k] > target && nums[k] > 0 && target > 0){
                return res;
            }
            if (k > 0 && nums[k] == nums[k-1]){
                continue;
            }
            for(int i = k + 1; i < nums.length; i++){
                if (nums[i] > target && nums[i] > 0 && target > 0){
                    break;
                }
                if (i > k + 1 && nums[i] == nums[i - 1]){
                    continue;
                }
                int left = i + 1, right = nums.length - 1;
                while(left < right){
                    long sum = 0L + nums[k] + nums[i] + nums[left] + nums[right];
                    if (sum > target) {
                        right--;
                    } else if(sum < target) {
                        left++;
                    } else {
                        res.add(Arrays.asList(nums[k], nums[i], nums[left], nums[right]));
                        left++;
                        right--;
                        while(left < right && nums[left] == nums[left - 1]) left++;
                        while(left < right && nums[right] == nums[right + 1])right--;
                    }
                }
            }
        }
        return res;
    }
}