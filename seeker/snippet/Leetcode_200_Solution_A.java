//date: 2023-10-23T16:39:46Z
//url: https://api.github.com/gists/08ce9b4b3f43b2c1121387fb2dfdec37
//owner: https://api.github.com/users/TheGreatJoules

class Solution {
    public int numIslands(char[][] grid) {
        int result = 0;
        boolean[][] visited = new boolean[grid.length][grid[0].length];
        Queue<int[]> queue = new LinkedList<>();
        for (int row = 0; row < grid.length; row++) {
            for (int col = 0; col < grid[0].length; col++) {
                if (grid[row][col] == '1' && !visited[row][col]) {
                    visited[row][col] = true;
                    queue.offer(new int[]{row, col});
                    bfs(grid, visited, queue);
                    result++;
                }
            }
        }
        return result;
    }

    private void bfs(char[][] grid, boolean[][] visited, Queue<int[]> queue) {
        int[][] dirs = new int[][] {{0,1}, {0,-1}, {1,0}, {-1, 0}};
        while (!queue.isEmpty()) {
            int[] current = queue.poll();
            for (int[] dir : dirs) {
                int x = current[0] + dir[0];
                int y = current[1] + dir[1];
                if (x >= 0 && x < grid.length && y >= 0 && y < grid[0].length && !visited[x][y] && grid[x][y] == '1') {
                    visited[x][y] = true;
                    queue.offer(new int[] {x, y});
                }
            }
        }
    }
}