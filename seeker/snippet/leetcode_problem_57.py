#date: 2023-04-14T17:07:19Z
#url: https://api.github.com/gists/2ba41a0e843facf19b113c859fc75a5c
#owner: https://api.github.com/users/qwertyvipul

# https://leetcode.ca/2016-01-26-57-Insert-Interval/

class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        start = newInterval[0]
        end = newInterval[1]

        left = list(filter(lambda interval: interval[1] < start, intervals))
        right = list(filter(lambda interval: interval[0] > end, intervals))

        if left + right != intervals:
            start = min(intervals[len(left)][0], start)
            end = max(intervals[~len(right)][1], end)
        return left + [[start, end]] + right