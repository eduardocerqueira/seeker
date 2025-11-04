//date: 2025-11-04T16:59:35Z
//url: https://api.github.com/gists/904d2e00f9914dcfe59d9cb170b0d969
//owner: https://api.github.com/users/aCoderOfTheSevenKingdoms

import java.io.*;
import java.util.*;

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

class Main{

    public static void main(String[] args){

        // Create a new Disjoint Set
        DisjointSet ds = new DisjointSet(7);

        // Add nodes & edges
        ds.unionByRank(1,2);
        ds.unionByRank(2,3);
        ds.unionByRank(4,5);
        ds.unionByRank(6,7);
        ds.unionByRank(5,6);

        // Check if 3 & 7 belong to the same component
        if(ds.findUltimateParent(3) == ds.findUltimateParent(7)){
            System.out.println("Same Component");
        } else {
            System.out.println("Different Component");
        }

        // Do a union of 3 & 7
        ds.unionByRank(3,7);

        if(ds.findUltimateParent(3) == ds.findUltimateParent(7)){
            System.out.println("Same Component");
        } else {
            System.out.println("Different Component");
        }
    }
}