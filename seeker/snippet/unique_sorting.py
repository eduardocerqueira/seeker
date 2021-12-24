#date: 2021-12-24T16:36:45Z
#url: https://api.github.com/gists/c246b94314b1aecd18865fd476cd448c
#owner: https://api.github.com/users/kaveryanovGFL

def uniqueSorting(nums: list) -> list:
    max_item = max(nums)
    indices = [nums.index(n) for n in range(1, max_item + 1)]
    return [n for _, n in sorted(zip(indices,range(1, max_item + 1)))]
result = uniqueSorting([4, 2, 1, 1, 2, 1, 1, 3, 2])