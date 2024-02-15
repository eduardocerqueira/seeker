#date: 2024-02-15T16:52:20Z
#url: https://api.github.com/gists/34e27f8ca467c0b762650725dae46ff0
#owner: https://api.github.com/users/elmarguseinov

def find_path(edges, a, b):
    graph = {}
    for edge in edges:
        u, v = edge
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)

    visited = set()

    def dfs(node, target, path):
        if node == target:
            return path
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                L = [node,neighbor]
                L.sort()
                L = tuple(L)
                new_path = dfs(neighbor, target, path + [L])
                if new_path:
                    return new_path
        return []

    return dfs(a, b, [])

###

def berge_minimal_hitting_set(hypergraph):
    # Initialize an empty hitting set
    hitting_set = set()

    # Iterate over all possible subsets of vertices
    for vertices_subset in powerset(get_all_vertices(hypergraph)):
        # If the current subset forms a hitting set
        if is_hitting_set(vertices_subset, hypergraph):
            # If hitting_set is empty or the current hitting set is smaller
            if not hitting_set or len(vertices_subset) < len(hitting_set):
                hitting_set = vertices_subset

    return len(hitting_set)

def get_all_vertices(hypergraph):
    # Returns a set containing all vertices in the hypergraph
    return set(vertex for hyperedge in hypergraph for vertex in hyperedge)

def is_hitting_set(vertices, hypergraph):
    # Checks if the given set of vertices forms a hitting set for the hypergraph
    for hyperedge in hypergraph:
        # If any hyperedge is not covered by vertices, return False
        if not any(vertex in vertices for vertex in hyperedge):
            return False
    return True

def powerset(s):
    # Generates all possible subsets of a set
    result = [[]]
    for elem in s:
        result.extend([x + [elem] for x in result])
    return result

###

t = int(input())
for experiment in range(t):
    n = int(input())
    Edges = []
    for i in range(n - 1):
        edge = [int(a) for a in input().split()]
        Edges.append(edge)
    k = int(input())
    Pairs = []
    for i in range(k):
        pair = [int(a) for a in input().split()]
        Pairs.append(pair)
    Paths = []
    for pair in Pairs:
        path = find_path(Edges, pair[0], pair[1])
        Paths.append(path)
    #print(Paths)
    print(berge_minimal_hitting_set(Paths))