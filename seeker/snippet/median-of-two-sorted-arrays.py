#date: 2023-05-11T16:53:41Z
#url: https://api.github.com/gists/23479f0a6e4e43903f66e1ba1908e258
#owner: https://api.github.com/users/markbrutx

from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        m, n = len(nums1), len(nums2)
        left, right = 0, m
        half_len = (m + n + 1) // 2

        while left <= right:
            i = (left + right) // 2
            j = half_len - i

            if i < m and nums2[j-1] > nums1[i]:
                left = i + 1
            elif i > 0 and nums1[i-1] > nums2[j]:
                right = i - 1
            else:
                if i == 0:
                    max_left = nums2[j-1]
                elif j == 0:
                    max_left = nums1[i-1]
                else:
                    max_left = max(nums1[i-1], nums2[j-1])

                if (m + n) % 2 == 1:
                    return max_left

                if i == m:
                    min_right = nums2[j]
                elif j == n:
                    min_right = nums1[i]
                else:
                    min_right = min(nums1[i], nums2[j])

                return (max_left + min_right) / 2.0

        raise ValueError("Input arrays are not sorted!")