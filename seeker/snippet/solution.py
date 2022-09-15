#date: 2022-09-15T17:10:00Z
#url: https://api.github.com/gists/d0a5e9b5cd183f5c0ea1b9b243305714
#owner: https://api.github.com/users/mateidanut

class Solution:
    def countGoodTriplets(self, arr: List[int], a: int, b: int, c: int) -> int:
        count = 0
        
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                for k in range(j+1, len(arr)):
                    
                    if (abs(arr[i] - arr[j]) <= a and
                        abs(arr[j] - arr[k]) <= b and
                        abs(arr[i] - arr[k]) <= c):
                        
                        count += 1
                        
        return count