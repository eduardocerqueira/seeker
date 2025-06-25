//date: 2025-06-25T17:13:57Z
//url: https://api.github.com/gists/a6b54ac157d30ede3eac855370c49fb8
//owner: https://api.github.com/users/ParasRaba155

// Graph is a generic graph with adjacency list representation.
// K must be comparable (for use as a map key), V can be any type.
type Graph[K comparable, V any] struct {
	adj map[K][]V
}

// NewGraph returns a new instance of a generic graph.
func NewGraph[K comparable, V any]() *Graph[K, V] {
	return &Graph[K, V]{
		adj: make(map[K][]V),
	}
}

// AddEdge adds an undirected edge between u and v.
// K and V must be the same type in this simple usage scenario.
func (g *Graph[K, V]) AddEdge(u K, v V) {
	g.adj[u] = append(g.adj[u], v)
}

// BFS performs a breadth-first traversal from a starting node key.
func (g *Graph[K, V]) BFS(start K, visit func(K)) {
	visited := make(map[K]bool)
	queue := []K{start}
	visited[start] = true

	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]

		visit(node)

		for _, neighbor := range g.adj[node] {
			neighborKey, ok := any(neighbor).(K)
			if !ok {
				continue // skip if not castable (safety net)
			}
			if !visited[neighborKey] {
				visited[neighborKey] = true
				queue = append(queue, neighborKey)
			}
		}
	}
}

// DFS performs a depth-first traversal using an explicit stack
func (g *Graph[K, V]) DFS(start K, visit func(K)) {
	visited := make(map[K]bool)
	stack := []K{start}

	for len(stack) > 0 {
		n := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if visited[n] {
			continue
		}
		visited[n] = true
		visit(n)

		for i := len(g.adj[n]) - 1; i >= 0; i-- {
			neighbor := g.adj[n][i]
			nk, ok := any(neighbor).(K)
			if !ok || visited[nk] {
				continue
			}
			stack = append(stack, nk)
		}
	}
}
