#date: 2026-02-26T17:39:26Z
#url: https://api.github.com/gists/09f676538a6c4fc42dfc7fdf15ba6518
#owner: https://api.github.com/users/nitflame

from collections import deque

def bfs(graph, start, goal):
    queue, visited = deque([(start, [start])]), {start}

    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        for n in graph.get(node, []):
            if n not in visited:
                visited.add(n)
                queue.append((n, path + [n]))
    return None


def main():
    graph = {}
    n = int(input("Number of nodes: "))

    for _ in range(n):
        node = input("Node: ").upper()
        graph[node] = input(f"Neighbors of {node}: ").upper().split()

    start = input("\nStart node: ").upper()
    goal = input("Goal node: ").upper()

    print("\nGraph:")
    for k, v in graph.items():
        print(f"{k} -> {' '.join(v)}")

    path = bfs(graph, start, goal)

    print("\nResult:")
    print(" -> ".join(path) if path else "No path found")


if __name__ == "__main__":
    main()