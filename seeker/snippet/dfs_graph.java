//date: 2022-07-20T17:05:04Z
//url: https://api.github.com/gists/a7203b52187c4e02b9f2e83051e5ec0b
//owner: https://api.github.com/users/imsurajmishra

import java.util.*;
import java.util.stream.IntStream;

public class GraphTraversal {

    static Set<Integer> visited = new HashSet<>();
    static List<List<Integer>> adjList = new ArrayList<>();
    static List<Integer> result = new ArrayList<>();
     
    public static void main(String[] args) {
        init(); // generate adjacency list
        dfs(8, adjList);
        result.forEach(r-> System.out.println("dfs traversed node : "+r));
    }
  
    
    public static void dfs(int v, List<List<Integer>> adjList){
        for(int i=1;i<v;i++){
            if(visited.contains(i)) {
                continue;
            }
            result.add(i);
            visited.add(i);
            dfs(i);
        }
    }


    private static void dfs(int node){
        if(visited.contains(node)) return;
        result.add(node);
        visited.add(node);
        List<Integer> adjNodes = adjList.get(node);
        for(int eachNode: adjNodes){
            dfs(node);
        }
    }
  
    private static void init() {
        // init
        IntStream.rangeClosed(0,5).forEach(i->adjList.add(new ArrayList<>()));
        adjList.add(1, Arrays.asList(2));
        adjList.add(2, Arrays.asList(1,3,7));
        adjList.add(3, Arrays.asList(2,5));
        adjList.add(4, Arrays.asList(6));
        adjList.add(5, Arrays.asList(3,7));
        adjList.add(6, Arrays.asList(4));
        adjList.add(7, Arrays.asList(2,5));
    }
}