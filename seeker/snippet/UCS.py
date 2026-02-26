#date: 2026-02-26T17:39:26Z
#url: https://api.github.com/gists/09f676538a6c4fc42dfc7fdf15ba6518
#owner: https://api.github.com/users/nitflame

import heapq

def ucs(graph, start, goal):
    pq, visited = [(0, start, [start])], set()

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node == goal:
            return path, cost

        if node not in visited:
            visited.add(node)
            for n, w in graph.get(node, []):
                if n not in visited:
                    heapq.heappush(pq, (cost + w, n, path + [n]))

    return None, float('inf')


def main():
    graph = {}
    n = int(input("Number of nodes: "))

    for _ in range(n):
        node = input("Node: ").upper()
        edges = input(f"Neighbors (node cost) of {node}: ").split()
        graph[node] = [(edges[i].upper(), int(edges[i+1])) for i in range(0, len(edges), 2)]

    start = input("\nStart node: ").upper()
    goal = input("Goal node: ").upper()

    print("\nGraph:")
    for k, v in graph.items():
        print(f"{k} -> {v}")

    path, cost = ucs(graph, start, goal)

    print("\nResult:")
    print("Path:", " -> ".join(path) if path else "No path found")
    print("Total cost:", cost if path else "∞")


if __name__ == "__main__":
    main()