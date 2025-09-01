//date: 2025-09-01T17:04:55Z
//url: https://api.github.com/gists/311b51d168aa2481c27254a558980229
//owner: https://api.github.com/users/kousthubha-sky

//485. Max Consecutive Ones
class Solution {
    public int findMaxConsecutiveOnes(int[] nums) {
        int n = nums.length;
        int cnt = 0,maxi = 0;
        for(int  i = 0;i<n;i++){
            if(nums[i]==1){
                cnt++;
                maxi = Math.max(maxi,cnt);
            }else{
                cnt = 0;
            }    
        }
        return maxi;
    }
}