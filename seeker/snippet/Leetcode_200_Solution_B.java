//date: 2023-10-23T16:39:46Z
//url: https://api.github.com/gists/08ce9b4b3f43b2c1121387fb2dfdec37
//owner: https://api.github.com/users/TheGreatJoules

class Solution {
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int result = 0;
        boolean[][] visited = new boolean[grid.length][grid[0].length];
        for (int row = 0; row < grid.length; row++) {
            for (int col = 0; col < grid[0].length; col++) {
                if (grid[row][col] == '1' && !visited[row][col]) {
                    dfs(grid, visited, row, col);
                    result++;
                }
            }
        }
        return result;
    }

    private void dfs(char[][] grid, boolean[][] visited, int row, int col) {
        if (row < 0 || row >= grid.length || col < 0 || col >= grid[0].length || visited[row][col] || grid[row][col] == '0') {
            return;
        }
        int[][] dirs = new int[][] {{0,1}, {0,-1}, {1,0}, {-1,0}};
        visited[row][col] = true;
        for (int[] dir : dirs) {
            dfs(grid, visited, row + dir[0], col + dir[1]);
        }
    }
}