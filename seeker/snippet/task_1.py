#date: 2025-08-13T16:40:41Z
#url: https://api.github.com/gists/c16e305a437ee7787571bd8dbb39ee8c
#owner: https://api.github.com/users/DanielIvanov19

class NumberSet:
    def __init__(self, n):
        if n > 64:
            raise ValueError("n should be ≤ 64")
        self.n = n
        self.bits = 0  # Initial empty

    def add(self, num):
        if 0 <= num <= self.n:
            self.bits |= (1 << num)

    def remove(self, num):
        if 0 <= num <= self.n:
            self.bits &= ~(1 << num)

    def contains(self, num):
        if 0 <= num <= self.n:
            return (self.bits >> num) & 1 == 1
        return False

    def __repr__(self):
        return "{" + ", ".join(str(i) for i in range(self.n + 1) if self.contains(i)) + "}"
