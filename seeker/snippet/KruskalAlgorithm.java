//date: 2025-11-05T17:06:55Z
//url: https://api.github.com/gists/3cfcb34e9845ca9eac7208606ca6637d
//owner: https://api.github.com/users/aCoderOfTheSevenKingdoms

// User function Template for Java
class DisjointSet{

    int[] size;
    int[] rank;
    int[] parent;

    public DisjointSet(int n){

        // Ranks for every node initialized to 0
        rank = new int[n+1];

        size = new int[n+1];
        // Size for each node initialized to 1
        Arrays.fill(size, 1);

        parent = new int[n+1];
        
        // Every node is the parent of itself initially
        for(int i = 0; i <= n; i++){
            parent[i] = i;
        }
    }

    public int findUltimateParent(int node){
        
        // Base case
        if(node == parent[node]) return node;

        // Path Compression 
        return parent[node] = findUltimateParent(parent[node]);
    }

    public void unionByRank(int u, int v){
        
        // Get the ultimate parents of u & v
        int ultimateParentU = findUltimateParent(u);
        int ultimateParentV = findUltimateParent(v);

        // If both the nodes belong to the same component no need to do union
        if(ultimateParentU == ultimateParentV) return;

        if(rank[ultimateParentU] < rank[ultimateParentV]){
            // Connect ultimate parent of u to ultimate parent of v. No need to update the rank of v.
            parent[ultimateParentU] = ultimateParentV;
        }

        else if(rank[ultimateParentU] > rank[ultimateParentV]){
            parent[ultimateParentV] = ultimateParentU;
        }

        // If both the ultimate parents have the same rank. Increase rank of either of the ultimate parent.
        else{
            parent[ultimateParentV] = ultimateParentU;
            rank[ultimateParentU]++;
        }
    }

    public void unionBySize(int u, int v){

        int ulp_u = findUltimateParent(u);
        int ulp_v = findUltimateParent(v);

        if(ulp_u == ulp_v) return;

        if(size[ulp_u] < size[ulp_v]){
            parent[ulp_u] = ulp_v;
            size[ulp_v] += size[ulp_u];
        } 

        else{
            parent[ulp_v] = ulp_u;
            size[ulp_u] += size[ulp_v];
        }
    }
};

class Edge implements Comparable<Edge>{
  
    int src, dst, weight;
    
    public Edge(int _src, int _dst, int _wt){
        this.src = _src;
        this.dst = _dst;
        this.weight = _wt;
    }
    
    public int compareTo(Edge compareEdge){
        return this.weight - compareEdge.weight; 
    }
};

class Solution {
    static int kruskalsMST(int V, int[][] edges) {
        // code here
        List<Edge> edgesList = new ArrayList<Edge>();
        
        for(int i = 0; i < edges.length; i++){
            
            int u = edges[i][0], v = edges[i][1], w = edges[i][2];
            edgesList.add(new Edge(u, v, w));
        }
        
        DisjointSet ds = new DisjointSet(V);
        
        Collections.sort(edgesList);
        
        int MSTsum = 0;
        
        for(Edge edge : edgesList){
            
            int u = edge.src, v = edge.dst, w = edge.weight;
            
            if(ds.findUltimateParent(u) != ds.findUltimateParent(v)){
                MSTsum += w;
                ds.unionBySize(u,v);
            }
        }
        
        return MSTsum;
    }
}
