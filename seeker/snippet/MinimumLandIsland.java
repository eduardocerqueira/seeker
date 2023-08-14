//date: 2023-08-14T16:42:13Z
//url: https://api.github.com/gists/9fad5f782ebbd62476f3f2a526206c1d
//owner: https://api.github.com/users/pushoo-sharma

package Strike.Tree.treeTraversal;

public class IslandCount {

    public int islandCount(char[][] grid) {
        
        int rows = grid[0].length;
        int cols = grid.length;
        int minCount = Integer.MAX_VALUE;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (grid[i][j] == 'L') {
                    int count = dfs(grid, i, j);
                    minCount = Math.min(minCount, count);
                }
            }
        }

        return minCount;

    }

    public int dfs(char[][] grid, int i, int j) {

        // base codition
        if(i < 0 || i >= grid[0].length || j < 0 || j >= grid.length || grid[i][j] != 'L') {
            return 0;
        }

        grid[i][j] = 'V';

        return 1 + dfs(grid, i - 1, j) +  dfs(grid, i + 1, j) + dfs(grid, i, j - 1) + dfs(grid, i, j + 1);

    }


    public static void main(String[] args) {
        
        IslandCount islandCounter = new IslandCount();
        
        char[][] grid = {
            {'L', 'L', 'W', 'W'},
            {'L', 'W', 'W', 'L'},
            {'W', 'L', 'W', 'L'},
            {'L', 'L', 'W', 'W'}
        };

        int islands = islandCounter.islandCount(grid);
        
        System.out.println("Number of islands: " + islands);

    }
    
}
