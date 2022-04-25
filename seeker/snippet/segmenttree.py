#date: 2022-04-25T16:56:24Z
#url: https://api.github.com/gists/016f9a2a9d6193492ba3847f2eb3fadc
#owner: https://api.github.com/users/orchardpark

class SegmentTree:
    """
    Segment tree in Python.
    Current implementation is used to find the sum
    over a given range in an array, but can be modified
    to perform other range queries.
    """

    def __init__(self, number_list):
        self.number_list = number_list
        self.tree = [0] * 4 * len(self.number_list)
        self.n = len(self.number_list)
        self.construct_tree()

    def construct_tree(self):
        self.build(1, 0, self.n - 1)

    def build(self, v, tl, tr):
        if tl == tr:
            self.tree[v] = self.number_list[tl]
        else:
            tm = (tl + tr) // 2
            self.build(v * 2, tl, tm)
            self.build(v * 2 + 1, tm + 1, tr)
            self.tree[v] = self.tree[v * 2] + self.tree[v * 2 + 1]

    def query_sum(self, left, right):
        return self.sum(1, 0, self.n - 1, left, right)

    def sum(self, v, tl, tr, l, r):
        if l > r:
            return 0
        if l == tl and r == tr:
            return self.tree[v]
        tm = (tl + tr) // 2
        return self.sum(v * 2, tl, tm, l, min(r, tm)) + \
            self.sum(v * 2 + 1, tm + 1, tr, max(l, tm+1), r)