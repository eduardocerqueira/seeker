#date: 2022-05-18T16:52:51Z
#url: https://api.github.com/gists/a3d77310e6b1e2023296c6b7642ca669
#owner: https://api.github.com/users/theabbie

from collections import defaultdict

class Solution:
    def DFS(self, graph, node, v, s):
        v.add(node)
        for j in graph[node]:
            if j not in v:
                self.DFS(graph, j, v, s)
        s.append(node)
    
    def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
        graph = defaultdict(list)
        transpose = defaultdict(list)
        for a, b in connections:
            graph[a].append(b)
            transpose[b].append(a)
        s = []
        self.DFS(graph, 0, set(), s)
        while len(s) > 0:
            curr = s.pop()
            v = set()
            self.DFS(transpose, curr, v, [])
            print(v)