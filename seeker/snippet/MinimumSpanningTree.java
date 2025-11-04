//date: 2025-11-04T16:59:35Z
//url: https://api.github.com/gists/904d2e00f9914dcfe59d9cb170b0d969
//owner: https://api.github.com/users/aCoderOfTheSevenKingdoms

class Solution {
    public int spanningTree(int V, int[][] edges) {
        // code here
        List<List<int[]>> adjList = new ArrayList<>();
        
        for(int i = 0; i < V; i++){
            adjList.add(new ArrayList<>());
        }
        
        for(int i = 0; i < edges.length; i++){
            
            int u = edges[i][0], v = edges[i][1], w = edges[i][2];
            adjList.get(u).add(new int[]{v, w});
            adjList.get(v).add(new int[]{u, w});
        }
        
        PriorityQueue<int[]> minheap = new PriorityQueue<>((x,y) -> {
            return x[0] - y[0];
        });
        
        minheap.offer(new int[]{0, 0});
        
        boolean[] visited = new boolean[V];
        
        int MSTsum = 0;
        
        while(!minheap.isEmpty()){
            
            int[] node = minheap.poll();
            
            int currWeight = node[0], currNode = node[1];
            
            if(visited[currNode]) continue;
            
            visited[currNode] = true;
            
            MSTsum += currWeight;
            
            for(int[] adj : adjList.get(currNode)){
                
                int adjNode = adj[0], adjWeight = adj[1];
                
                if(!visited[adjNode]){
                    minheap.offer(new int[]{adjWeight, adjNode});
                }
            }
        }
        
        return MSTsum;
    }
}
