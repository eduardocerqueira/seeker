#date: 2024-02-16T16:54:53Z
#url: https://api.github.com/gists/416fb9a968fe0deef147e0e14dfb5666
#owner: https://api.github.com/users/rlid

def match(s, p):
    m = len(s)
    n = len(p)

    @cache
    def f(i, j):
        if i == m:
            while j < n:
                if p[j] != '*':
                    return False
                j += 1
            return True
        elif j == n:
            return False

        if p[j] != '*':
            if p[j] == '?' or s[i] == p[j]:
                return f(i + 1, j + 1)
            else:
                return False
        else:
            return f(i + 1, j) or f(i, j + 1)

    return f(0, 0)



class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        return match(s, p)