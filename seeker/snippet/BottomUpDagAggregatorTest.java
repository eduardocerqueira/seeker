//date: 2025-02-27T16:58:52Z
//url: https://api.github.com/gists/47c8e1b0901e0b1fe806c299617d1e6e
//owner: https://api.github.com/users/rbezzi

import org.jgrapht.Graph;
import org.jgrapht.graph.DefaultDirectedGraph;
import org.jgrapht.graph.DefaultEdge;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class BottomUpDagAggregatorTest {

    private final BottomUpDagAggregator aggregator = new BottomUpDagAggregator();
    private final GraphVizExporter exporter = new GraphVizExporter();

    @Test
    void testExample1() throws IOException {
        // Given this DAG:
        // A1 -> B1 -> B2
        // A2 -> C1 -> C2
        // A3 -> C3 -> C4
        // A4 -> B3 -> B4
        Graph<String, DefaultEdge> graph = createGraph1();

        // Export original graph
        exporter.exportToSVG(graph, "example1_input.svg");

        // Aggregate by property (first letter)
        Graph<BottomUpDagAggregator.VertexGroup<String, Character>, DefaultEdge> result =
                aggregator.aggregateByProperty(graph, vertex -> vertex.charAt(0));

        // Export result graph
        exporter.exportAggregatedGraphToSVG(result, "example1_output.svg");

        // Assert
        assertEquals(3, result.vertexSet().size()); // Should have 3 groups: A, B, and C

        // Find groups
        BottomUpDagAggregator.VertexGroup<String, Character> groupA = findGroupByProperty(result, 'A');
        BottomUpDagAggregator.VertexGroup<String, Character> groupB = findGroupByProperty(result, 'B');
        BottomUpDagAggregator.VertexGroup<String, Character> groupC = findGroupByProperty(result, 'C');

        assertNotNull(groupA);
        assertNotNull(groupB);
        assertNotNull(groupC);

        // Check vertices in each group
        assertEquals(4, groupA.getVertices().size());
        assertTrue(groupA.getVertices().containsAll(List.of("A1", "A2", "A3", "A4")));

        assertEquals(4, groupB.getVertices().size());
        assertTrue(groupB.getVertices().containsAll(List.of("B1", "B2", "B3", "B4")));

        assertEquals(4, groupC.getVertices().size());
        assertTrue(groupC.getVertices().containsAll(List.of("C1", "C2", "C3", "C4")));

        // Check edges between groups
        assertTrue(result.containsEdge(groupA, groupB));
        assertTrue(result.containsEdge(groupA, groupC));
        assertFalse(result.containsEdge(groupB, groupC));
        assertFalse(result.containsEdge(groupC, groupB));

        // Ensure no cycles exist
        assertNoCycles(result);
    }

    @Test
    void testExample2() throws IOException {
        // Given this DAG:
        // A1 -> B1 -> B2
        // A2 -> C1 -> C2 -> B2
        // A3 -> C3 -> C4
        // A4 -> B3 -> B4
        Graph<String, DefaultEdge> graph = createGraph2();

        // Export original graph
        exporter.exportToSVG(graph, "example2_input.svg");

        // Aggregate by property (first letter)
        Graph<BottomUpDagAggregator.VertexGroup<String, Character>, DefaultEdge> result =
                aggregator.aggregateByProperty(graph, vertex -> vertex.charAt(0));

        // Export result graph
        exporter.exportAggregatedGraphToSVG(result, "example2_output.svg");

        // Assert
        assertEquals(3, result.vertexSet().size()); // Should have 3 groups: A, B, and C

        // Find groups
        BottomUpDagAggregator.VertexGroup<String, Character> groupA = findGroupByProperty(result, 'A');
        BottomUpDagAggregator.VertexGroup<String, Character> groupB = findGroupByProperty(result, 'B');
        BottomUpDagAggregator.VertexGroup<String, Character> groupC = findGroupByProperty(result, 'C');

        assertNotNull(groupA);
        assertNotNull(groupB);
        assertNotNull(groupC);

        // Check vertices in each group
        assertEquals(4, groupA.getVertices().size());
        assertTrue(groupA.getVertices().containsAll(List.of("A1", "A2", "A3", "A4")));

        assertEquals(4, groupB.getVertices().size());
        assertTrue(groupB.getVertices().containsAll(List.of("B1", "B2", "B3", "B4")));

        assertEquals(4, groupC.getVertices().size());
        assertTrue(groupC.getVertices().containsAll(List.of("C1", "C2", "C3", "C4")));

        // Check edges between groups
        assertTrue(result.containsEdge(groupA, groupB));
        assertTrue(result.containsEdge(groupA, groupC));
        assertTrue(result.containsEdge(groupC, groupB));
        assertFalse(result.containsEdge(groupB, groupC));

        // Ensure no cycles exist
        assertNoCycles(result);
    }

    @Test
    void testExample3() throws IOException {
        // Given this DAG:
        // A1 -> B1 -> B2
        // A2 -> C1 -> C2 -> B2
        // A3 -> C3 -> C4
        // A4 -> B3 -> B4, B3 -> C4
        Graph<String, DefaultEdge> graph = createGraph3();

        // Export original graph
        exporter.exportToSVG(graph, "example3_input.svg");

        // Aggregate by property (first letter)
        Graph<BottomUpDagAggregator.VertexGroup<String, Character>, DefaultEdge> result =
                aggregator.aggregateByProperty(graph, vertex -> vertex.charAt(0));

        // Export result graph
        exporter.exportAggregatedGraphToSVG(result, "example3_output.svg");

        // Assert: The expected groups are A, B12, B34, and C1234
        assertEquals(4, result.vertexSet().size());

        // Find A group - should be all A vertices
        BottomUpDagAggregator.VertexGroup<String, Character> groupA = findGroupByProperty(result, 'A');
        assertNotNull(groupA);
        assertEquals(4, groupA.getVertices().size());
        assertTrue(groupA.getVertices().containsAll(List.of("A1", "A2", "A3", "A4")));

        // Find B groups - should be split into B12 and B34
        List<BottomUpDagAggregator.VertexGroup<String, Character>> bGroups = findGroupsByProperty(result, 'B');
        assertEquals(2, bGroups.size());

        BottomUpDagAggregator.VertexGroup<String, Character> groupB12 = null;
        BottomUpDagAggregator.VertexGroup<String, Character> groupB34 = null;

        for (BottomUpDagAggregator.VertexGroup<String, Character> group : bGroups) {
            if (group.getVertices().contains("B1") && group.getVertices().contains("B2")) {
                groupB12 = group;
            } else if (group.getVertices().contains("B3") && group.getVertices().contains("B4")) {
                groupB34 = group;
            }
        }

        assertNotNull(groupB12);
        assertNotNull(groupB34);

        assertEquals(2, groupB12.getVertices().size());
        assertTrue(groupB12.getVertices().containsAll(List.of("B1", "B2")));

        assertEquals(2, groupB34.getVertices().size());
        assertTrue(groupB34.getVertices().containsAll(List.of("B3", "B4")));

        // Find C group - should have all C vertices
        BottomUpDagAggregator.VertexGroup<String, Character> groupC = findGroupByProperty(result, 'C');
        assertNotNull(groupC);
        assertEquals(4, groupC.getVertices().size());
        assertTrue(groupC.getVertices().containsAll(List.of("C1", "C2", "C3", "C4")));

        // Check edges between groups
        assertTrue(result.containsEdge(groupA, groupB12));
        assertTrue(result.containsEdge(groupA, groupB34));
        assertTrue(result.containsEdge(groupA, groupC));
        assertTrue(result.containsEdge(groupC, groupB12));
        assertTrue(result.containsEdge(groupB34, groupC));

        // Ensure no cycles exist
        assertNoCycles(result);
    }

    /**
     * Ensures there are no cycles in the graph.
     */
    private <V> void assertNoCycles(Graph<V, DefaultEdge> graph) {
        for (V v1 : graph.vertexSet()) {
            for (V v2 : graph.vertexSet()) {
                if (!v1.equals(v2)) {
                    assertFalse(
                            graph.containsEdge(v1, v2) && graph.containsEdge(v2, v1),
                            "Cycle detected: " + v1 + " -> " + v2 + " -> " + v1
                    );
                }
            }
        }
    }

    private <V, P> BottomUpDagAggregator.VertexGroup<V, P> findGroupByProperty(
            Graph<BottomUpDagAggregator.VertexGroup<V, P>, ?> graph, P property) {

        for (BottomUpDagAggregator.VertexGroup<V, P> group : graph.vertexSet()) {
            if (group.getProperty().equals(property)) {
                return group;
            }
        }

        return null;
    }

    private <V, P> List<BottomUpDagAggregator.VertexGroup<V, P>> findGroupsByProperty(
            Graph<BottomUpDagAggregator.VertexGroup<V, P>, ?> graph, P property) {

        List<BottomUpDagAggregator.VertexGroup<V, P>> result = new ArrayList<>();

        for (BottomUpDagAggregator.VertexGroup<V, P> group : graph.vertexSet()) {
            if (group.getProperty().equals(property)) {
                result.add(group);
            }
        }

        return result;
    }

    private Graph<String, DefaultEdge> createGraph1() {
        Graph<String, DefaultEdge> graph = new DefaultDirectedGraph<>(DefaultEdge.class);

        // Add vertices
        for (String vertex : List.of("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4")) {
            graph.addVertex(vertex);
        }

        // Add edges
        graph.addEdge("A1", "B1");
        graph.addEdge("B1", "B2");
        graph.addEdge("A2", "C1");
        graph.addEdge("C1", "C2");
        graph.addEdge("A3", "C3");
        graph.addEdge("C3", "C4");
        graph.addEdge("A4", "B3");
        graph.addEdge("B3", "B4");

        return graph;
    }

    private Graph<String, DefaultEdge> createGraph2() {
        Graph<String, DefaultEdge> graph = new DefaultDirectedGraph<>(DefaultEdge.class);

        // Add vertices
        for (String vertex : List.of("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4")) {
            graph.addVertex(vertex);
        }

        // Add edges
        graph.addEdge("A1", "B1");
        graph.addEdge("B1", "B2");
        graph.addEdge("A2", "C1");
        graph.addEdge("C1", "C2");
        graph.addEdge("C2", "B2");
        graph.addEdge("A3", "C3");
        graph.addEdge("C3", "C4");
        graph.addEdge("A4", "B3");
        graph.addEdge("B3", "B4");

        return graph;
    }

    private Graph<String, DefaultEdge> createGraph3() {
        Graph<String, DefaultEdge> graph = new DefaultDirectedGraph<>(DefaultEdge.class);

        // Add vertices
        for (String vertex : List.of("A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4")) {
            graph.addVertex(vertex);
        }

        // Add edges
        graph.addEdge("A1", "B1");
        graph.addEdge("B1", "B2");
        graph.addEdge("A2", "C1");
        graph.addEdge("C1", "C2");
        graph.addEdge("C2", "B2");
        graph.addEdge("A3", "C3");
        graph.addEdge("C3", "C4");
        graph.addEdge("A4", "B3");
        graph.addEdge("B3", "B4");
        graph.addEdge("B3", "C4");

        return graph;
    }
}
