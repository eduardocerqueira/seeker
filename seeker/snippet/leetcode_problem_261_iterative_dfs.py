#date: 2023-07-03T16:45:21Z
#url: https://api.github.com/gists/3520c79d4b117707dccc31ec015f78d3
#owner: https://api.github.com/users/qwertyvipul

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) != n - 1: return False

        adjList = [[] for _ in range(n)]
        for A, B in edges:
            adjList[A].append(B)
            adjList[B].append(A)

        parent = {0: -1}
        stack = [0]

        while stack:
            node = stack.pop()
            for neighbour in adjList[node]:
                if neighbour == parent[node]:
                    continue

                if neighbour in parent:
                    return False

                parent[neighbour] = node
                stack.append(neighbour)

        return len(parent) == n