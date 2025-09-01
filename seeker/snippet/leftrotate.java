//date: 2025-09-01T17:04:55Z
//url: https://api.github.com/gists/311b51d168aa2481c27254a558980229
//owner: https://api.github.com/users/kousthubha-sky

// 169 rotate array
class Solution {
    void reverse(int[] arr, int low, int high) {
        int temp;
        while (low <= high) {
            temp = arr[low];
            arr[low] = arr[high];
            arr[high] = temp;
            low++;
            high--;
        }
    }

    public void rotate(int[] nums, int k) {
        int n = nums.length;
        k = k % n;
        reverse(nums, 0, n - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, n - 1);
    }
}