//date: 2021-10-05T17:02:38Z
//url: https://api.github.com/gists/09c534ac37a46656815b58fe4a70e360
//owner: https://api.github.com/users/ysyS2ysy

import java.util.*;
// 2021-10-06 02:00
class Solution {
    public int findDuplicate(int[] nums) {
        // nums 안에는 반복되는 숫자가 오로지 단 한개뿐이다.
        // 1. Map 객체에 저장하여 중복체크 (하위10%, 41ms)
        /**
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; i++){
            if(!map.containsKey(nums[i])){
                map.put(nums[i], 1);
            }else{
                return nums[i];
            }
        } // end of for
        */
        
        // 2. 배열에 저장하여 중복체크 (하위 43%, 9ms)
        boolean[] check = new boolean[100001];
        for(int i = 0; i < nums.length; i++){
            if(!check[nums[i]]){
                check[nums[i]] = true;
            }else{
                return nums[i];
            }
        }
        return -1;
    } // end of findDuplicate
} // end of class