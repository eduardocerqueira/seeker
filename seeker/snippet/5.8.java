//date: 2025-12-22T16:53:00Z
//url: https://api.github.com/gists/0fe3818f2aea43f7b7f36d0b4ba6ce77
//owner: https://api.github.com/users/JasonRon123

class Solution743 {
    public int networkDelayTime(int[][] times, int n, int k) {
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int i = 1; i <= n; i++) {
            graph.put(i, new ArrayList<>());
        }
        for (int[] time : times) {
            graph.get(time[0]).add(new int[]{time[1], time[2]});
        }
        
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> a[1] - b[1]);
        pq.offer(new int[]{k, 0});
        
        Map<Integer, Integer> dist = new HashMap<>();
        while (!pq.isEmpty()) {
            int[] current = pq.poll();
            int node = current[0];
            int time = current[1];
            
            if (dist.containsKey(node)) continue;
            dist.put(node, time);
            
            for (int[] neighbor : graph.get(node)) {
                if (!dist.containsKey(neighbor[0])) {
                    pq.offer(new int[]{neighbor[0], time + neighbor[1]});
                }
            }
        }
        
        if (dist.size() != n) return -1;
        return Collections.max(dist.values());
    }
}