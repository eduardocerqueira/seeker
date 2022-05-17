#date: 2022-05-17T16:48:17Z
#url: https://api.github.com/gists/ff1413c78d73d8a6f45bc22b752404fa
#owner: https://api.github.com/users/twhi

class Solution:
    def firstBadVersion(self, n) -> int:
        left, right = 1, n
        while left < right:
            mid = left + (right - left) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid + 1
        return left