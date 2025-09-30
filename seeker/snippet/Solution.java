//date: 2025-09-30T17:06:59Z
//url: https://api.github.com/gists/4e3d30e63876146bd5f7ecfa63f6640a
//owner: https://api.github.com/users/tanmaydubey1947

class Solution {

  class DisjointSet {

    private int[] rank;
    private int[] size;
    public int[] parent;

    public DisjointSet(int n) {
      rank = new int[n + 1];
      parent = new int[n + 1];
      size = new int[n + 1];
      Arrays.fill(size, 1);

      for (int i = 0; i <= n; i++) {
        parent[i] = i;
      }
    }

    public int findUPar(int node) {
      if (node == parent[node]) {
        return node;
      }
      return parent[node] = findUPar(parent[node]);
    }

    public boolean find(int u, int v) {
      return findUPar(u) == findUPar(v);
    }

    public void unionByRank(int u, int v) {

      int ulp_u = findUPar(u);
      int ulp_v = findUPar(v);

      if (ulp_u == ulp_v) {
        return;
      }

      if (rank[ulp_u] > rank[ulp_v]) {
        parent[ulp_v] = ulp_u;
      } else if (rank[ulp_u] < rank[ulp_v]) {
        parent[ulp_u] = parent[ulp_v];
      } else {
        parent[ulp_u] = ulp_v;
        rank[ulp_v]++;
      }
    }

    public void unionBySize(int u, int v) {
      int ulp_u = findUPar(u);
      int ulp_v = findUPar(v);

      if (ulp_u == ulp_v) {
        return;
      }

      if (size[ulp_u] < size[ulp_v]) {
        parent[ulp_u] = ulp_v;
        size[ulp_v] += size[ulp_u];
      } else {
        parent[ulp_v] = ulp_u;
        size[ulp_u] += size[ulp_v];
      }
    }
  }

  public int solve(int n, int[][] Edge) {

    DisjointSet ds = new DisjointSet(n);
    int extraEdges = 0;
    int m = Edge.length;

    for (int i = 0; i < m; i++) {

      int u = Edge[i][0];
      int v = Edge[i][1];

      if (ds.findUPar(u) == ds.findUPar(v)) {
        extraEdges++;
      } else {
        ds.unionByRank(u, v);
      }
    }

    int component = 0;

    for (int i = 0; i < n; i++) {
      if (ds.parent[i] == i) {
        component++;
      }
    }

    int ans = component - 1;

    return extraEdges >= ans ? ans : -1;
  }
}