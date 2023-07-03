#date: 2023-07-03T16:45:21Z
#url: https://api.github.com/gists/3520c79d4b117707dccc31ec015f78d3
#owner: https://api.github.com/users/qwertyvipul

class UnionFind:
    def __init__(self, n):
        self.parent = [node for node in range(n)]
        self.size = [1] * n

    def find(self, A):
        root = A
        while root != self.parent[root]:
            root = self.parent[root]

        while A != root:
            oldRoot = self.parent[A]
            self.parent[A] = root
            A = oldRoot
        return root

    def union(self, A, B):
        rootA = self.find(A)
        rootB = self.find(B)

        if rootA == rootB:
            return False

        if self.size[rootA] < self.size[rootB]:
            self.parent[rootA] = rootB
            self.size[rootB] += self.size[rootA]
        else:
            self.parent[rootB] = rootA
            self.size[rootA] += self.size[rootB]
        return True

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) != n - 1: return False

        unionFind = UnionFind(n)

        for A, B in edges:
            if not unionFind.union(A, B):
                return False

        return True