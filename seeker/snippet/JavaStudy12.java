//date: 2023-06-26T16:53:02Z
//url: https://api.github.com/gists/3fa2079d66c17e21867174633dcf588f
//owner: https://api.github.com/users/youngvctr

import java.util.ArrayList;
import java.util.HashMap;
import java.util.PriorityQueue;
import java.util.Arrays;

class Solution {
  /*
    dijkstra zb 강의 내용 활용
  */
    static ArrayList<ArrayList<Node>> graph;
    static HashMap<Integer, Integer> map = new HashMap<>();
    final static int INF = 50000;    
        
    static class Node{ 
        int to;
        int weight;
        
        public Node(int to, int weight){
            this.to = to;
            this.weight = weight;
        }
    }
    
    public static int dijkstra(int n, int start, int end){
        PriorityQueue<Node> pq = new PriorityQueue<>((x, y) -> x.weight - y.weight);
        pq.offer(new Node(start, 0));
        
        int[] dist = new int[n+1];
        for(int i=0; i< n+1; i++){
            dist[i] = INF;
        }
        dist[start] = 0;
        boolean[] visited = new boolean[n+1];
        
        while(!pq.isEmpty()){
            Node curNode = pq.poll();
            
            if(visited[curNode.to]){
                continue;
            }
            visited[curNode.to] = true;
        
            for(int i=0; i<graph.get(curNode.to).size(); i++) {
                Node adjNode = graph.get(curNode.to).get(i);
                if(!visited[adjNode.to] && dist[adjNode.to] > curNode.weight + adjNode.weight){
                    dist[adjNode.to] = curNode.weight + adjNode.weight;
                    pq.offer(new Node(adjNode.to, dist[adjNode.to]));
                }
            }
        }
        
        for(int i=1; i<dist.length; i++){
            map.put(i, dist[i]);
        }
        return dist[end];
    }
    
    public int solution(int n, int[][] edge) {
        int answer = 0;
        graph = new ArrayList<>();
        
        for(int i=0; i<n+1; i++) graph.add(new ArrayList<>());
        for(int i=0; i<edge.length; i++) {
            graph.get(edge[i][0]).add(new Node(edge[i][1], 1));
            graph.get(edge[i][1]).add(new Node(edge[i][0], 1));
        }
        
        dijkstra(n, 1, n);
        
        int max = Integer.MIN_VALUE;
        for(int i=0; i<=n; i++){
            if(map.get(i)!= null){
                if(max < map.get(i)){
                    max = map.get(i);
                }
            }
        }
        
        for(int i=0; i<=n; i++){
            if(map.get(i)!= null){
                if(max == map.get(i)){
                    answer++;
                }
            }
        }
        
        return answer;
    }
}