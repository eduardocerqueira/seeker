//date: 2022-05-24T17:03:43Z
//url: https://api.github.com/gists/7ddcdf0810187b219f4f1344dcba2ad7
//owner: https://api.github.com/users/mathi2001

package org.mycodes;
import java.util.*;
public class main {
    public int[] num(int[] nums, int target) {
        HashMap<Integer, Integer> hMap = new HashMap<>();
        int arr[] = new int[2];
        for (int i = 0; i < nums.length; i++) {
            int m = target - nums[i];
            if (hMap.containsKey(m)) {
                arr[0] = hMap.get(m);
                arr[1] = i;
                return arr;
            }
            hMap.put(nums[i],i);
        }
        return arr;
    }
}