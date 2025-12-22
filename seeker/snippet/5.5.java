//date: 2025-12-22T16:50:09Z
//url: https://api.github.com/gists/1e262c66e4cb04afb8716cc9a542972b
//owner: https://api.github.com/users/JasonRon123

import java.util.*;

public class Experiment {
    public static void main(String[] args) {
        Random rand = new Random();
        
        Graph[] graphs = new Graph[5];
        for (int i = 0; i < 5; i++) {
            graphs[i] = generateGraph(1000 * (i + 1), 5000 * (i + 1));
        }
        
        System.out.println("BFS vs DFS сравнение:");
        for (int i = 0; i < 5; i++) {
            Graph g = graphs[i];
            int startVertex = new ArrayList<>(g.getVertices()).get(0);
            
            long start = System.nanoTime();
            g.bfs(startVertex);
            long bfsTime = System.nanoTime() - start;
            
            start = System.nanoTime();
            g.dfs(startVertex);
            long dfsTime = System.nanoTime() - start;
            
            System.out.printf("Граф %d: BFS=%d ns, DFS=%d ns\n", 
                i + 1, bfsTime, dfsTime);
        }
        
        Graph weightedGraph = generateWeightedGraph(100, 500);
        Graph unweightedGraph = convertToUnweighted(weightedGraph);
        
        int start = new ArrayList<>(weightedGraph.getVertices()).get(0);
        
        Dijkstra dijkstra = new Dijkstra();
        long startTime = System.nanoTime();
        Map<Integer, Integer> dijkstraResult = dijkstra.dijkstra(weightedGraph, start);
        long dijkstraTime = System.nanoTime() - startTime;
        
        startTime = System.nanoTime();
        List<Integer> bfsResult = weightedGraph.bfs(start);
        long bfsTime = System.nanoTime() - startTime;
        
        System.out.println("\nДейкстра vs BFS:");
        System.out.println("Время Дейкстры: " + dijkstraTime + " ns");
        System.out.println("Время BFS: " + bfsTime + " ns");
        
        int testVertex = new ArrayList<>(weightedGraph.getVertices()).get(10);
        System.out.println("Расстояние до вершины " + testVertex + ":");
        System.out.println("Дейкстра: " + dijkstraResult.get(testVertex));
        
        GreedyAlgorithms greedy = new GreedyAlgorithms();
        int[][] intervals = {
            {1, 3}, {2, 5}, {4, 6}, {6, 8}, {7, 9}
        };
        
        int result = greedy.activitySelection(intervals);
        System.out.println("\nЖадный алгоритм:");
        System.out.println("Максимальное количество непересекающихся интервалов: " + result);
        
        int[][] intervals2 = {
            {1, 2}, {3, 4}, {5, 6}, {7, 8}
        };
        int result2 = greedy.activitySelection(intervals2);
        System.out.println("Оптимальное решение для непересекающихся интервалов: " + result2);
    }
    
    static Graph generateGraph(int vertices, int edges) {
        Graph g = new Graph();
        Random rand = new Random();
        
        for (int i = 0; i < vertices; i++) {
            g.addVertex(i);
        }
        
        for (int i = 0; i < edges; i++) {
            int v1 = rand.nextInt(vertices);
            int v2 = rand.nextInt(vertices);
            if (v1 != v2) {
                g.addEdge(v1, v2);
            }
        }
        
        return g;
    }
    
    static Graph generateWeightedGraph(int vertices, int edges) {
        Graph g = new Graph();
        Random rand = new Random();
        
        for (int i = 0; i < vertices; i++) {
            g.addVertex(i);
        }
        
        for (int i = 0; i < edges; i++) {
            int v1 = rand.nextInt(vertices);
            int v2 = rand.nextInt(vertices);
            int weight = rand.nextInt(10) + 1;
            if (v1 != v2) {
                g.addEdge(v1, v2, weight);
            }
        }
        
        return g;
    }
    
    static Graph convertToUnweighted(Graph weighted) {
        Graph g = new Graph();
        
        for (int vertex : weighted.getVertices()) {
            g.addVertex(vertex);
            for (Graph.Edge edge : weighted.getNeighbors(vertex)) {
                g.addEdge(vertex, edge.vertex);
            }
        }
        
        return g;
    }
}