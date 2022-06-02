//date: 2022-06-02T17:03:45Z
//url: https://api.github.com/gists/7449138481f14417dc7b8eec3c0d3503
//owner: https://api.github.com/users/vinodhalaharvi

import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

class Node {
    private final int id;
    Set<Node> adj;
    boolean visited;

    Node(int id) {
        this.id = id;
    }

    public static Node create(int id) {
        return new Node(id);
    }

    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Node)) return false;
        Node node = (Node) o;
        return getId() == node.getId();
    }

    @Override public String toString() {
        return "Node{" +
                "id=" + id +
                ", adj=" + adj +
                ", visited=" + visited +
                '}';
    }

    @Override public int hashCode() {
        return Objects.hash(getId());
    }

    public int getId() {
        return id;
    }

    public Set<Node> getAdj() {
        return adj;
    }

    public boolean isVisited() {
        return visited;
    }

    public void setVisited(boolean visited) {
        this.visited = visited;
    }
}

class Graph {
    static Set<Node> allNodes = new HashSet<>();

    public static void add(Node node, Set<Node> adj) {
        node.adj = adj;
        allNodes.add(node);
    }

    public static Graph create() {
        return new Graph();
    }

    public static void bfs(Node node) {
        if (node != null) {
            if (!isVisited(node)) {
                visit(node);
            }
            if (node.getAdj() != null) {
                node.getAdj().forEach(Graph::bfs);
            }
        }
    }

    private static void visit(Node node) {
        node.setVisited(true);
        System.out.println("node = " + node);
    }

    private static boolean isVisited(Node node) {
        return node.isVisited();
    }

    public static void dfs(Node node) {

    }

    public Set<Node> getAllNodes() {
        return allNodes;
    }
}


class GraphTest {
    public static void main(String[] args) {
        HashSet<Node> nodes = new HashSet<>();
        nodes.add(Node.create(2));
        Graph.add(Node.create(1), nodes);

        nodes.clear();
        nodes.add(Node.create(3));
        nodes.add(Node.create(4));
        Graph.add(Node.create(2), nodes);

        nodes.clear();
        nodes.add(Node.create(5));
        nodes.add(Node.create(6));
        Graph.add(Node.create(3), nodes);

        Graph.bfs(Node.create(2));
    }
}