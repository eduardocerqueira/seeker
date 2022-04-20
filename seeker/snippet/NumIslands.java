//date: 2022-04-20T17:17:25Z
//url: https://api.github.com/gists/9e69b2ca11085ceed9aaa578f78aa638
//owner: https://api.github.com/users/temirlan100

import java.util.LinkedList;
import java.util.Queue;

public class NumIslands {
    public static int numIslands(char[][] grid) {
//        if (grid == null || grid.length == 0) {
//            return 0;
//        }

        int neighborRow = grid.length;
        int neighborColumn = grid[0].length;
        var numIslands = 0;

        for (var row = 0; row < neighborRow; row++) {
            for (var col = 0; col < neighborColumn; col++) {

                if (grid[row][col] == '1') {
                    numIslands++;

                    grid[row][col] = '0';

                    Queue<Integer> neighbors = new LinkedList<>();
                    neighbors.add(row * neighborColumn + col);

                    while (!neighbors.isEmpty()) {
                        int id = neighbors.poll();
                        int rowIsland = id / neighborColumn;
                        int columnIsland = id % neighborColumn;

                        if (rowIsland - 1 >= 0 && grid[rowIsland - 1][columnIsland] == '1') {
                            neighbors.add((rowIsland - 1) * neighborColumn + columnIsland);
                            grid[rowIsland - 1][columnIsland] = '0';
                        }

                        if (rowIsland + 1 < neighborRow && grid[rowIsland + 1][columnIsland] == '1') {
                            neighbors.add((rowIsland + 1) * neighborColumn + columnIsland);
                            grid[rowIsland + 1][columnIsland] = '0';
                        }

                        if (columnIsland - 1 >= 0 && grid[rowIsland][columnIsland - 1] == '1') {
                            neighbors.add(rowIsland * neighborColumn + columnIsland - 1);
                            grid[rowIsland][columnIsland - 1] = '0';
                        }

                        if (columnIsland + 1 < neighborColumn && grid[rowIsland][columnIsland + 1] == '1') {
                            neighbors.add(rowIsland * neighborColumn + columnIsland + 1);
                            grid[rowIsland][columnIsland + 1] = '0';
                        }

                    }
                }
            }
        }

        return numIslands;
    }

    public static void main(String[] args) {
        char[][] grid = {
                {'1', '1', '0', '0', '0'},
                {'1', '1', '0', '0', '0'},
                {'0', '0', '1', '0', '0'},
                {'0', '0', '0', '1', '1'}
        };

        System.out.println(numIslands(grid));
    }
}
