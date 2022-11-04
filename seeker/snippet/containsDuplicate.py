#date: 2022-11-04T16:51:16Z
#url: https://api.github.com/gists/9d35df265be66892d625f17d7bfae17e
#owner: https://api.github.com/users/madaniel

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        counter = set()
        
        for number in nums:
            if number in counter:
                return True
            counter.add(number)
        
        return False