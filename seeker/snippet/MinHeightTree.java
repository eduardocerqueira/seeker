//date: 2024-12-24T16:44:34Z
//url: https://api.github.com/gists/0851d893ca22de22114b1fb8ab9fa11e
//owner: https://api.github.com/users/vedesh-padal

class Solution {

    // INTUITION:
        // to get the min. height, find the diameter of the graph (tree)
        // the middle node is your root of the min. height tree
        // if diameter of tree is of length:
            // odd: middle node in the diameter path is the only root node 
            //      that is possible that forms the min. tree hight
            // even: there are two root nodes possible that form the min. tree height

        // so, how do we find the diameter?
        // here there are two appraoches:
        // 1st approach:
        // as discussed in codestoryWithMIK video: [ VERY IMPORTANT CONCEPT ]
            // 1. pick a node, find the farthest node (that will definitely be the one end of the diameter)
            // 2. now, from this one end, find the farthest node - in the process you have the diameter length
            // your problem solved
        // slightly different approach: 
        // 2nd approach - kinda easier: topological sort


    // Based on Topological sort approach
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        // but this is undirected graph (becoz bidirection edges) - so how?
        
        // in DAG, we used to do BFS (Queue) starting from the 0 degree nodes
        // since, acc. to this question, it is not directed, 
        // we pick the start nodes as those which have least indegree, i.e., 1
        // ONION PEELING APPROACH - TOPO SORT

        // edge case: n = 1
        if (n == 1) {
            // new thing learnt
            return Collections.singletonList(0);
        }

        // indegree
        int[] indegree = new int[n];
        // Graph
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] edge: edges) {
            indegree[edge[0]]++;
            indegree[edge[1]]++;
            graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            graph.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }

        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            if (indegree[i] == 1) {
                q.offer(i);
            }
        }

        int processed = 0;
        
        // ALSO, height can also be obtained becoz of the level wise processing
        int height = 0;

        // since the min. height of the tree root can be in 1 or 2 in number, 
        // hence the condition
        while (n - processed > 2) {
            int size = q.size();
            processed += size;
            for (int i = 0; i < size; i++) {
                int curr = q.poll();
                for (int neighbor: graph.get(curr)) {
                    // athi uttam way of writing code, top notch
                    if (--indegree[neighbor] == 1) {
                        q.offer(neighbor);
                    }
                }
            }
            height++;
        }

        List<Integer> res = new ArrayList<>();
        res.addAll(q);
        return res;
    }


    // =================== initial try =========================================

    private int levelOrder(int root, List<Integer>[] graph) {
        boolean[] vis = new boolean[graph.length];
        Queue<Integer> q = new LinkedList<>();
        q.offer(root);
        vis[root] = true;
        
        int levels = 0;    // here, since the graph is a tree, levels here means => height of the tree
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                int curr = q.poll();
                for (int neighbor: graph[curr]) {
                    if (!vis[neighbor]) {
                        vis[neighbor] = true;
                        q.offer(neighbor);
                    }
                }
            }
            levels++;
        }
        return levels;
    }

    public List<Integer> findMinHeightTrees_TLE(int n, int[][] edges) {
        // SIMPLE: --- EVIL LAUGH FROM LEETCODE -> says: TLE (think better kid) - my try
        // do a level order traversal (LOT) from each node, and find 
        // no. of levels when starting the LOT from that node
        // keep track of these levels in a hashmap, with key -> levels, value -> root node
        // find the min. no. of level from the hashmap, and return the root nodes's values stored 

        // first construct the graph
        List<Integer>[] graph = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            graph[i] = new ArrayList<>();
        }

        for (int[] edge: edges) {
            graph[edge[0]].add(edge[1]);
            graph[edge[1]].add(edge[0]);
        }

        Map<Integer, List<Integer>> lvlAndRoots = new HashMap<>();

        // now, do a level order traversal and count the levels (height) if 
        // each node is taken as the root and done the traversal
        for (int i = 0; i < n; i++) {
            int height = levelOrder(i, graph);
            lvlAndRoots.computeIfAbsent(height, k -> new ArrayList<>()).add(i);
        }

        System.out.println(lvlAndRoots);

        int minHeight = Integer.MAX_VALUE;
        for (int key: lvlAndRoots.keySet()) {
            minHeight = Math.min(minHeight, key);
        }

        return lvlAndRoots.get(minHeight);
    }
}