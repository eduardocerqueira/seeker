#date: 2023-03-06T16:39:01Z
#url: https://api.github.com/gists/30f6b9992630d1399612f1fe74f9e2be
#owner: https://api.github.com/users/JustLiveKZ

class MinStack:

    def __init__(self):
        self.stack = []
        self.mins = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.mins.append(val if not self.mins else min(val, self.mins[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.mins.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.mins[-1]