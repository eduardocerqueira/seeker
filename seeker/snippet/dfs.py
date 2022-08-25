#date: 2022-08-25T15:18:26Z
#url: https://api.github.com/gists/4dda83fcfbadf61a9b463ff2aaf41453
#owner: https://api.github.com/users/0scarB

def dfs(graph, start, goal, path=None, visited=None):
    path = path or []
    visited = visited or set()

    path.append(start)
    visited.add(start)

    if start == goal:
        return path

    for neighbor in graph[start]:
        if neighbor not in visited:
            try:
                return dfs(graph, neighbor, goal, path, visited)
            except ValueError:
                pass

    raise ValueError(f"Could not find path from {start=} to {goal=}")


if __name__ == "__main__":
    path = dfs(
        {
            0: [2, 3, 4],
            1: [0, 6],
            2: [4, 6],
            3: [2, 4, 5, 6],
            4: [0, 1, 2, 3, 5, 6],
            5: [0, 1, 3, 4],
            6: [1, 2, 4, 5]
        },
        0,
        1
    )

    print(path)
