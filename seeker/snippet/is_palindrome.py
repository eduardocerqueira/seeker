#date: 2023-10-27T17:01:15Z
#url: https://api.github.com/gists/47a7fd33ed07b7463b41ac9280051019
#owner: https://api.github.com/users/mlivingston40

class Solution:
    def isPalindrome(self, x: int) -> bool:
        try:
            compare = self.constructRightToLeft(x)
        except ValueError:
            return False
        return x == compare

    def constructRightToLeft(self, x:int) -> int:
        out = ''
        for i in range(len(str(x))):
            i = -i - 1
            out += str(x)[i]
        return int(out)
