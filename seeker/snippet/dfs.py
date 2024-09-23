#date: 2024-09-23T16:56:08Z
#url: https://api.github.com/gists/d7aea5b11fbb6a52357e76c395f0a455
#owner: https://api.github.com/users/RobKwiatkowski

visited_nodes = set()

def dfs(visited: set, graph: dict, node: str) -> None:
    """
    Deep First Search - writing to an external set and printing nodes
    Args:
        visited: a set where all visited nodes will be writen to
        graph: graph as an adjacency matrix
        node: a starting node for a search

    Returns: None
    """
    if node not in visited:
        print(node)
        visited.add(node)
        for neigh in graph[node]:
            dfs(visited, graph, neigh)
    return None