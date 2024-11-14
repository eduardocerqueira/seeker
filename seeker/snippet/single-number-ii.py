#date: 2024-11-14T17:07:34Z
#url: https://api.github.com/gists/c3f0a8dca11db563e6088a530b121d04
#owner: https://api.github.com/users/motyzk

class Solution:
    def singleNumber(self, nums):
        pos_ret_val = 0
        neg_ret_val = 0
        for bit in range(32):
            pos_counter = 0
            neg_counter = 0
            for num in nums:
                if num > 0:
                    if (num & (1 << bit)):
                        pos_counter += 1
                else:
                    if (-num & (1 << bit)):
                        neg_counter += 1
            pos_ret_val += (pos_counter % 3) * (1 << bit)
            neg_ret_val += (neg_counter % 3) * (1 << bit)
        return pos_ret_val - neg_ret_val
