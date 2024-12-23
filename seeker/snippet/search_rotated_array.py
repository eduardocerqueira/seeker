#date: 2024-12-23T16:48:35Z
#url: https://api.github.com/gists/ac181c04d83b532d7e75277f47538f7d
#owner: https://api.github.com/users/jymchng

from typing import List

def search_rotated_array(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # Determine which side is sorted
        if nums[left] <= nums[mid]:  # Left side is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # Right side is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1