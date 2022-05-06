#date: 2022-05-06T17:15:29Z
#url: https://api.github.com/gists/341630203aed27cd7d78ca8296a0a3f2
#owner: https://api.github.com/users/hassansadiq1

class Solution:
    def __init__(self) -> None:
        self.map = {
            "I": 1,
            "V": 5,
            "X": 10,
            "L": 50,
            "C": 100,
            "D": 500,
            "M": 1000
        }
        self.map_order = {
            "I": 1,
            "V": 2,
            "X": 3,
            "L": 4,
            "C": 5,
            "D": 6,
            "M": 7
        }

    def romanToInt(self, s: str) -> int:
        rs = s[::-1]
        length = len(rs)
        value = 0
        for i in range(length):
            if (i+1) < length:
                if self.map_order[rs[i]] > self.map_order[rs[i+1]]:
                    value += self.map[rs[i]] - 2*self.map[rs[i+1]]
                else:
                    value += self.map[rs[i]]
            else:
                value += self.map[rs[i]]
        return value
