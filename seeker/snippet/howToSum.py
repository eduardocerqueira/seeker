#date: 2022-05-05T17:16:20Z
#url: https://api.github.com/gists/233c62c6eb42bc9561118981a891f319
#owner: https://api.github.com/users/fribas84

from typing import List

def howToSum(target: int, nums: List[int], init = False, memo = {}) -> List[int]:
    if(init == False): memo.clear()  # the reason of this is because how dictionaries work in python, the object will remain in memory
    if (target in memo): return memo[target]
    if(target == 0): return []
    if(target <0): return None


    for num in nums:
        remainder = target - num
        remRes = howToSum(remainder, nums,True, memo)
        if (remRes != None):
            remRes.append(num)
            memo[target] = remRes
            return remRes
    memo[target] = None
    return None        



print(howToSum(7,[2,3])) 
print(howToSum(7,[5,3,4,7])) 
print(howToSum(7,[2,4])) 
print(howToSum(8,[2,3,5])) 
print(howToSum(300,[7,14])) 


