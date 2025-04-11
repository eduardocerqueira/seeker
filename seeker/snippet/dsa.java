//date: 2025-04-11T16:43:30Z
//url: https://api.github.com/gists/22e4ef208e8d2db460f22c2027a04bcb
//owner: https://api.github.com/users/leet-somnath

import java.util.*;

public class Solution {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int[] arr = new int[n];
        for (int i = 0; i < n; i++) {
            arr[i] = sc.nextInt();
        }
        int target = sc.nextInt();
        int[] result = solve(arr, target);
        System.out.println(Arrays.toString(result));
    }

    static int[] solve(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (map.containsKey(complement)) {
                return new int[]{map.get(complement), i};
            }
            map.put(nums[i], i);
        }
        return new int[]{}; // No solution (though problem states one exists)
    }
}
