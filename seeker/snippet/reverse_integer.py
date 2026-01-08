#date: 2026-01-08T17:23:09Z
#url: https://api.github.com/gists/8cce62c1575896b75c91ea00988a9750
#owner: https://api.github.com/users/iamvalson

class Solution:
    def reverse(self, x: int) -> int:

        INT_MAX =  2147483647
        INT_MIN = -2147483648

        sign = 1

        if x < 0:
            sign = -1
        x = abs(x)
        reverse = 0
        while x > 0:
            digit = x % 10
            reverse = reverse * 10 + digit
            x = x // 10

        reverse *= sign
        if reverse < INT_MIN or reverse > INT_MAX:
            return 0
        return reverse
            

        return output

        