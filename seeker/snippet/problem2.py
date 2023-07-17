#date: 2023-07-17T16:58:55Z
#url: https://api.github.com/gists/1a2ff46122bd398e31571096e54f43fc
#owner: https://api.github.com/users/AsifAlFaisal

from collections import Counter

def find_single_value(nums):
    result = Counter(nums)
    for k, v in result.items():
        if v==1:
            return k

if __name__=="__main__":

    nums = [2, 2, 1]
    print(find_single_value(nums))

    nums = [4,1,2,1,2]
    print(find_single_value(nums))

    nums = [1]
    print(find_single_value(nums))