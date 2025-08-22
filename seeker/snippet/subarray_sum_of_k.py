#date: 2025-08-22T17:14:31Z
#url: https://api.github.com/gists/abc8a0a03f4848af5b719828ffdbabc7
#owner: https://api.github.com/users/Lumberj3ck

# Given an array of integers and an integer target, find a subarray that sums to target and return the start and end indices of the subarray.
#
# Input: arr: 1 -20 -3 30 5 4 target: 7
#
# Output: 1 4
# Explanation: -20 - 3 + 30 = 7. The indices for subarray [-20,-3,30] is 1 and 4 (right exclusive)
def subarray_of_k(nums=[1,2, 3, 1, 8, 1,2,3] , target = 15):
    curr_sum = 0
    prefix = {0:0}

    for i in range(len(nums)):
        curr_sum += nums[i] # 7 

        # 7 
        complement = curr_sum - target


        if complement in prefix:
            return [prefix[complement], i]

        prefix[curr_sum] = i + 1

    return [-1, -1]

