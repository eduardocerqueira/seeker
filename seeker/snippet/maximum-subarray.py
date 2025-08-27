#date: 2025-08-27T17:13:35Z
#url: https://api.github.com/gists/fec48bbcec302696f391e5f438b4ac4c
#owner: https://api.github.com/users/motyzk

def maximum_subarray(nums):
    biggest_sum = nums[0]
    current_sum = 0
    for n in nums:
        current_sum += n
        if current_sum > biggest_sum:
            biggest_sum = current_sum
        if current_sum < 0:
            current_sum = 0
    return biggest_sum


assert maximum_subarray([-1]) == -1
assert maximum_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
assert maximum_subarray([-2, 1, 5, -3, 4, -1, 2, 1, -5, 4]) == 9
assert maximum_subarray([1]) == 1
assert maximum_subarray([-2, -1]) == -1
