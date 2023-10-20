#date: 2023-10-20T17:01:35Z
#url: https://api.github.com/gists/2402173beee640f6297f5638227e769d
#owner: https://api.github.com/users/jssonx

class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        if not nums1 or not nums2:
            return []
        pq = []
        for i in range(len(nums1)):
            heapq.heappush(pq, [nums1[i] + nums2[0], i, 0])
        res = []
        while pq and k > 0:
            cur_sum, i, j = heapq.heappop(pq)
            res.append([nums1[i], nums2[j]])
            k -= 1
            if j + 1 < len(nums2):
                heapq.heappush(pq, [nums1[i] + nums2[j+1], i, j+1]) 
        return res