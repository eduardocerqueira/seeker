#date: 2022-09-29T17:27:09Z
#url: https://api.github.com/gists/378a9aef7fa391f5a78637c360a75d78
#owner: https://api.github.com/users/mateidanut

class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and n & (n-1) == 0