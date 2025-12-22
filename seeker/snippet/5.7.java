//date: 2025-12-22T16:52:18Z
//url: https://api.github.com/gists/b4e86def6b700b1baf4258c75290a654
//owner: https://api.github.com/users/JasonRon123

class Solution207 {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        List<List<Integer>> graph = new ArrayList<>();
        for (int i = 0; i < numCourses; i++) {
            graph.add(new ArrayList<>());
        }
        for (int[] pre : prerequisites) {
            graph.get(pre[1]).add(pre[0]);
        }
        
        int[] visited = new int[numCourses];
        for (int i = 0; i < numCourses; i++) {
            if (hasCycle(graph, visited, i)) {
                return false;
            }
        }
        return true;
    }
    
    private boolean hasCycle(List<List<Integer>> graph, int[] visited, int course) {
        if (visited[course] == 1) return true;
        if (visited[course] == 2) return false;
        
        visited[course] = 1;
        for (int next : graph.get(course)) {
            if (hasCycle(graph, visited, next)) {
                return true;
            }
        }
        visited[course] = 2;
        return false;
    }
}