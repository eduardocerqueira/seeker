//date: 2024-09-16T17:10:28Z
//url: https://api.github.com/gists/12ac859a0f6d500e52d8ae7999e2b395
//owner: https://api.github.com/users/qren0neu

class Solution {
    public long maxScore(int[] nums1, int[] nums2, int k) {
        // if use priority queue, we can have:
        // 1. when we poll in the queue, we remove the min
        // so the sum of nums1 should be larger
        // but, we have to calculate the minimum dynamically in nums2
        // if we can combine nums1 and nums2 somehow together, we can solve the problem
        int[][] arr = new int[nums1.length][2];
        for (int i = 0; i < nums1.length; i++) {
            arr[i][0] = nums1[i];
            arr[i][1] = nums2[i];
        }
        Arrays.sort(arr, (int[] arr1, int[] arr2) -> arr2[1] - arr1[1]);
        PriorityQueue<Integer> pq = new PriorityQueue<Integer>(k, (a,b) -> a - b);
        long score = 0;
        long sum = 0;
        for (int[] pair : arr) {
            // pair: nums1, nums2
            int min = pair[1];
            pq.offer(pair[0]);
            sum += pair[0];
            if (pq.size() > k) {
                int removed = pq.poll();
                sum -= removed;
            }
            if (pq.size() == k) {
                score = Math.max(score, sum * min);
            }
        }
        return score;
    }
}