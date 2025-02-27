//date: 2025-02-27T16:58:52Z
//url: https://api.github.com/gists/47c8e1b0901e0b1fe806c299617d1e6e
//owner: https://api.github.com/users/rbezzi

import org.jgrapht.Graph;
import org.jgrapht.graph.AsSubgraph;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Queue;
import java.util.Set;
import java.util.function.Function;

/**
 * A class that aggregates vertices of a DAG based on their properties using a bottom-up approach.
 * Starts with each vertex in its own group and merges groups when possible.
 */
public class BottomUpDagAggregator {

    /**
     * Represents a group of vertices with the same property.
     */
    private static class Group<V, P> {
        private final P property;
        private final Set<V> vertices;

        public Group(P property, V initialVertex) {
            this.property = property;
            this.vertices = new HashSet<>();
            this.vertices.add(initialVertex);
        }

        public Group(P property, Collection<V> vertices) {
            this.property = property;
            this.vertices = new HashSet<>(vertices);
        }

        public P getProperty() {
            return property;
        }

        public Set<V> getVertices() {
            return Collections.unmodifiableSet(vertices);
        }

        public void addVertex(V vertex) {
            vertices.add(vertex);
        }

        public void addAll(Collection<V> newVertices) {
            vertices.addAll(newVertices);
        }

        public int size() {
            return vertices.size();
        }

        public boolean isEmpty() {
            return vertices.isEmpty();
        }

        @Override
        public String toString() {
            return property + ": " + vertices;
        }
    }

    /**
     * Represents a candidate for merging two groups.
     */
    private static class MergeCandidate<V, P> {
        private final Group<V, P> group1;
        private final Group<V, P> group2;
        private final int score;

        public MergeCandidate(Group<V, P> group1, Group<V, P> group2, int score) {
            this.group1 = group1;
            this.group2 = group2;
            this.score = score;
        }

        public Group<V, P> getGroup1() {
            return group1;
        }

        public Group<V, P> getGroup2() {
            return group2;
        }

        public int getScore() {
            return score;
        }
    }

    /**
     * Helper class to track path information during BFS.
     */
    private static class PathInfo<V> {
        final V vertex;
        final boolean passedThroughDifferentProperty;

        PathInfo(V vertex, boolean passedThroughDifferentProperty) {
            this.vertex = vertex;
            this.passedThroughDifferentProperty = passedThroughDifferentProperty;
        }
    }

    /**
     * Aggregates vertices of a DAG based on their properties according to the specified rules.
     * Uses a bottom-up approach, starting with each vertex in its own group and merging when possible.
     *
     * @param originalGraph The original DAG
     * @param propertyExtractor Function to extract property from a vertex
     * @param <V> Vertex type
     * @param <E> Edge type
     * @param <P> Property type
     * @return A new DAG where vertices are aggregated based on properties
     */
    public <V, E, P> Graph<VertexGroup<V, P>, DefaultEdge> aggregateByProperty(
            Graph<V, E> originalGraph,
            Function<V, P> propertyExtractor) {

        // Initialize: Each vertex starts in its own group
        List<Group<V, P>> groups = new ArrayList<>();
        Map<V, Group<V, P>> vertexToGroup = new HashMap<>();

        for (V vertex : originalGraph.vertexSet()) {
            P property = propertyExtractor.apply(vertex);
            Group<V, P> group = new Group<>(property, vertex);
            groups.add(group);
            vertexToGroup.put(vertex, group);
        }

        // Keep merging until no more merges are possible
        boolean merged = true;
        while (merged) {
            merged = false;

            // Find the best merge candidate
            MergeCandidate<V, P> bestCandidate = findBestMergeCandidate(
                    originalGraph, groups, vertexToGroup, propertyExtractor);

            if (bestCandidate != null) {
                // Perform the merge
                Group<V, P> mergedGroup = mergeGroups(bestCandidate.getGroup1(),
                        bestCandidate.getGroup2(), groups, vertexToGroup);
                merged = true;
            }
        }

        // Create the final result DAG
        return createResultDAG(originalGraph, groups, propertyExtractor);
    }

    /**
     * Finds the best candidate for merging groups.
     * Returns null if no valid merge candidates are found.
     */
    private <V, E, P> MergeCandidate<V, P> findBestMergeCandidate(
            Graph<V, E> graph,
            List<Group<V, P>> groups,
            Map<V, Group<V, P>> vertexToGroup,
            Function<V, P> propertyExtractor) {

        List<MergeCandidate<V, P>> candidates = new ArrayList<>();

        // Check all pairs of groups with the same property
        for (int i = 0; i < groups.size(); i++) {
            for (int j = i + 1; j < groups.size(); j++) {
                Group<V, P> group1 = groups.get(i);
                Group<V, P> group2 = groups.get(j);

                if (group1.isEmpty() || group2.isEmpty()) {
                    continue;
                }

                // Only consider groups with the same property
                if (group1.getProperty().equals(group2.getProperty())) {
                    // Check if these groups can be merged
                    if (canMergeGroups(graph, group1, group2, groups, vertexToGroup, propertyExtractor)) {
                        // Add to candidates with a score (total size)
                        int score = group1.size() + group2.size();
                        candidates.add(new MergeCandidate<>(group1, group2, score));
                    }
                }
            }
        }

        // Return the best candidate (highest score), or null if none found
        return candidates.isEmpty() ? null :
                candidates.stream()
                        .max(Comparator.comparingInt(MergeCandidate::getScore))
                        .orElse(null);
    }

    /**
     * Checks if two groups can be merged without creating paths through different properties.
     */
    private <V, E, P> boolean canMergeGroups(
            Graph<V, E> graph,
            Group<V, P> group1,
            Group<V, P> group2,
            List<Group<V, P>> allGroups,
            Map<V, Group<V, P>> vertexToGroup,
            Function<V, P> propertyExtractor) {

        P property = group1.getProperty();

        // Create an augmented graph that includes edges representing current group structure
        DefaultDirectedGraph<V, DefaultEdge> augmentedGraph =
                new DefaultDirectedGraph<>(DefaultEdge.class);

        // Add all vertices from the original graph
        for (V vertex : graph.vertexSet()) {
            augmentedGraph.addVertex(vertex);
        }

        // Add edges from the original graph
        for (V source : graph.vertexSet()) {
            for (E edge : graph.outgoingEdgesOf(source)) {
                V target = graph.getEdgeTarget(edge);
                augmentedGraph.addEdge(source, target);
            }
        }

        // Add edges to represent vertices already in the same group
        // (except for the groups being considered for merging)
        for (Group<V, P> g : allGroups) {
            if (g == group1 || g == group2) {
                continue;
            }

            List<V> groupVertices = new ArrayList<>(g.getVertices());
            for (int i = 0; i < groupVertices.size(); i++) {
                for (int j = 0; j < groupVertices.size(); j++) {
                    V v1 = groupVertices.get(i);
                    V v2 = groupVertices.get(j);
                    if (!v1.equals(v2)) {
                        augmentedGraph.addEdge(v1, v2);
                    }
                }
            }
        }

        // Check all pairs of vertices between the groups
        for (V v1 : group1.getVertices()) {
            for (V v2 : group2.getVertices()) {
                // Check if there's a path through a different property in either direction
                if (hasPathThroughDifferentProperty(augmentedGraph, v1, v2, property, propertyExtractor) ||
                        hasPathThroughDifferentProperty(augmentedGraph, v2, v1, property, propertyExtractor)) {
                    return false;
                }
            }
        }

        return true;  // Safe to merge
    }

    /**
     * Checks if there's a path from source to target that passes through a vertex with a different property.
     */
    private <V, P> boolean hasPathThroughDifferentProperty(
            Graph<V, DefaultEdge> graph,
            V source,
            V target,
            P sourceProperty,
            Function<V, P> propertyExtractor) {

        Set<V> visited = new HashSet<>();
        Queue<PathInfo<V>> queue = new LinkedList<>();

        visited.add(source);
        queue.add(new PathInfo<>(source, false));

        while (!queue.isEmpty()) {
            PathInfo<V> current = queue.poll();
            V currentVertex = current.vertex;
            boolean passedThroughDifferent = current.passedThroughDifferentProperty;

            for (DefaultEdge edge : graph.outgoingEdgesOf(currentVertex)) {
                V neighbor = graph.getEdgeTarget(edge);

                // If we reached the target, check if we passed through a different property
                if (neighbor.equals(target)) {
                    if (passedThroughDifferent) {
                        return true;  // Found a path through a different property
                    }
                    continue;  // Don't explore beyond target, but keep looking for other paths
                }

                if (!visited.contains(neighbor)) {
                    P neighborProperty = propertyExtractor.apply(neighbor);

                    // Check if this neighbor has a different property
                    boolean neighborPassedThrough = passedThroughDifferent ||
                            !sourceProperty.equals(neighborProperty);

                    visited.add(neighbor);
                    queue.add(new PathInfo<>(neighbor, neighborPassedThrough));
                }
            }
        }

        return false;  // No path through a different property
    }

    /**
     * Merges two groups and updates the groups list and vertex-to-group mapping.
     */
    private <V, P> Group<V, P> mergeGroups(
            Group<V, P> group1,
            Group<V, P> group2,
            List<Group<V, P>> groups,
            Map<V, Group<V, P>> vertexToGroup) {

        // Create a new merged group
        Group<V, P> mergedGroup = new Group<>(group1.getProperty(),
                new ArrayList<>());
        mergedGroup.addAll(group1.getVertices());
        mergedGroup.addAll(group2.getVertices());

        // Update the vertex-to-group mapping
        for (V vertex : mergedGroup.getVertices()) {
            vertexToGroup.put(vertex, mergedGroup);
        }

        // Update the groups list
        groups.remove(group1);
        groups.remove(group2);
        groups.add(mergedGroup);

        return mergedGroup;
    }

    /**
     * Creates the final result DAG.
     */
    private <V, E, P> Graph<VertexGroup<V, P>, DefaultEdge> createResultDAG(
            Graph<V, E> originalGraph,
            List<Group<V, P>> groups,
            Function<V, P> propertyExtractor) {

        Graph<VertexGroup<V, P>, DefaultEdge> resultGraph =
                new DefaultDirectedGraph<>(DefaultEdge.class);

        // Create a mapping from original vertices to their groups
        Map<V, VertexGroup<V, P>> vertexToResultGroup = new HashMap<>();

        // Create vertices in the result graph
        int groupCounter = 1;
        for (Group<V, P> group : groups) {
            if (group.isEmpty()) continue;

            // Create a subgraph for this group
            Graph<V, E> subgraph = new AsSubgraph<>(originalGraph, group.getVertices());

            // Create a vertex group
            String groupId = group.getProperty().toString() + groupCounter++;
            VertexGroup<V, P> vertexGroup = new VertexGroup<>(
                    groupId, group.getProperty(), group.getVertices(), subgraph);
            resultGraph.addVertex(vertexGroup);

            // Update mapping
            for (V vertex : group.getVertices()) {
                vertexToResultGroup.put(vertex, vertexGroup);
            }
        }

        // Create edges in the result graph
        for (V source : originalGraph.vertexSet()) {
            VertexGroup<V, P> sourceGroup = vertexToResultGroup.get(source);

            for (E edge : originalGraph.outgoingEdgesOf(source)) {
                V target = originalGraph.getEdgeTarget(edge);
                VertexGroup<V, P> targetGroup = vertexToResultGroup.get(target);

                // Add edge between groups if they are different
                if (!sourceGroup.equals(targetGroup) && !resultGraph.containsEdge(sourceGroup, targetGroup)) {
                    resultGraph.addEdge(sourceGroup, targetGroup);
                }
            }
        }

        return resultGraph;
    }

    /**
     * Represents a group of vertices in the result DAG.
     */
    public static class VertexGroup<V, P> {
        private final String id;
        private final P property;
        private final Set<V> vertices;
        private final Graph<V, ?> subgraph;

        public VertexGroup(String id, P property, Set<V> vertices, Graph<V, ?> subgraph) {
            this.id = id;
            this.property = property;
            this.vertices = Collections.unmodifiableSet(vertices);
            this.subgraph = subgraph;
        }

        public String getId() {
            return id;
        }

        public P getProperty() {
            return property;
        }

        public Set<V> getVertices() {
            return vertices;
        }

        public Graph<V, ?> getSubgraph() {
            return subgraph;
        }

        @Override
        public String toString() {
            return id + ": " + vertices;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            VertexGroup<?, ?> that = (VertexGroup<?, ?>) o;
            return Objects.equals(id, that.id);
        }

        @Override
        public int hashCode() {
            return Objects.hash(id);
        }
    }
}
