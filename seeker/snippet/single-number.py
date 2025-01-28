#date: 2025-01-28T17:06:31Z
#url: https://api.github.com/gists/f0ee4fd442f0983c32eb43bfd5788d26
#owner: https://api.github.com/users/motyzk

class Solution:
    def singleNumber(self, nums):
        s = set()
        for num in nums:
            if num not in s:
                s.add(num)
            else:
                s.remove(num)
        return s.pop()


class Solution:
    def singleNumber(self, nums):
        ret_val = 0
        for num in nums:
            # ret_val = ret_val ^ num
            ret_val ^= num
        return ret_val