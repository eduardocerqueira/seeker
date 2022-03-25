//date: 2022-03-25T17:14:40Z
//url: https://api.github.com/gists/d21bb4fe91ee36afe11bec80ed9e76d0
//owner: https://api.github.com/users/ErikDokken2

import java.awt.*;
import java.util.*;

public class maze {
    static Scanner stdin = new Scanner(System.in);
    static int[] rowNum = {-1, 0, 0, 1};
    static int[] colNum = {0, -1, 1, 0};
    static HashMap<Character, ArrayList<Point>> map = new HashMap<>();
    // A Data Structure for queue used in BFS
    static class queueNode{
        Point pt; // The coordinates of a cell
        int dist; // cell's distance of from the source

        public queueNode(Point pt, int dist)
        {
            this.pt = pt;
            this.dist = dist;
        }
    }

    public static void main(String[] args) {

        //Variables
        int mazeHeight = stdin.nextInt();
        int mazeWidth = stdin.nextInt();
        boolean[][] visitedGrid = new boolean[mazeWidth][mazeHeight];
        //Start and End Point
        Point startPoint = new Point(0, 0);
        Point endPoint = new Point(0, 0);

        char[][] grid = new char[mazeWidth][mazeHeight];




        //Gets MazeData (StartPoint and EndPoint and Makes the grid)
        getMazeData(grid,mazeHeight,mazeWidth, startPoint, endPoint, visitedGrid, map);

        //Solve
        int distance = solveMaze(grid, startPoint, endPoint, mazeHeight, mazeWidth, visitedGrid);

        System.out.println(distance);
        //System.out.print(map);
    }


    //Reads inputted maze
    private static void getMazeData(char[][] grid, int mazeHeight, int mazeWidth, Point source, Point dest, boolean[][] visitedGrid, HashMap<Character, ArrayList<Point>> map) {
        stdin.nextLine();
        for(int y = 0; y < mazeHeight; y++){
            String temp = stdin.nextLine();
            for(int x = 0; x < mazeWidth; x++){
                char charTemp = temp.charAt(x);

                if(charTemp != ('!') && charTemp != ('*') && charTemp != ('.') && charTemp != ('$')){
                    char key = charTemp;
                    Point pointTemp = new Point(x,y);
                    map.computeIfAbsent(key, k -> new ArrayList<>());
                    map.get(key).add(pointTemp);
                }

                grid[x][y] = (temp.charAt(x));

                //Sets up start and end
                if(charTemp == ('$')){
                    dest.x = x;
                    dest.y = y;
                }
                else if(charTemp == ('*')){
                    source.x = x;
                    source.y = y;
                }
                //sets all the walls as visited so its not processed during solve maze
                else if(charTemp == ('!')){
                    visitedGrid[x][y] = true;
                }
            }
        }
    }
    //BFS Function
    private static int solveMaze(char[][] grid, Point startPoint, Point endPoint, int mazeHeight, int mazeWidth, boolean[][] visitedGrid) {

        //Makes sure that the source and destination is correct
        if(grid[startPoint.x][startPoint.y] != ('*') || grid[endPoint.x][endPoint.y] != ('$')) {
            System.out.print("Error Start and End! :(");
            return -1;
        }


        //Make start point visited
        visitedGrid[startPoint.x][startPoint.y] = true;

        //Make a queue for the BFS where each queueNode holds the x and y as well as the distance
        Queue<queueNode> queue = new LinkedList<>();
        queueNode source = new queueNode(startPoint, 0);
        queue.add(source);

        //BFS Loop
        while(!queue.isEmpty()){
            //Current Value Testing
            queueNode currentCord = queue.peek();
            Point currentPoint = currentCord.pt;

            //Exit Found
            if(currentPoint.x == endPoint.x && currentPoint.y == endPoint.y){
                return currentCord.dist;
            }
            //Where is it hit a letter
            if(grid[currentPoint.x][currentPoint.y] != ('*') && grid[currentPoint.x][currentPoint.y] != ('!') && grid[currentPoint.x][currentPoint.y] != ('.') && grid[currentPoint.x][currentPoint.y] != ('$')){
                char key = grid[currentPoint.x][currentPoint.y];

                for(int x = 0; x < map.get(key).size(); x++){
                    int row = map.get(key).get(x).x;
                    int col = map.get(key).get(x).y;

                    if (isValid(row, col, mazeHeight,mazeWidth) && (grid[row][col] != ('!') && !visitedGrid[row][col]))
                    {
                        // mark cell as visited and enqueue it
                        visitedGrid[row][col] = true;
                        //adds one to the distance
                        queueNode Adjcell = new queueNode (new Point(row, col), currentCord.dist + 1 );
                        queue.add(Adjcell);
                    }
                }
            }
            queue.remove();

            //There is only 4 ways we can go up, down, right, left..... or teleport
            for(int x = 0; x < 4; x++){
                int row = currentPoint.x + rowNum[x];
                int col = currentPoint.y + colNum[x];

                // if adjacent cell is valid, has path
                // and not visited yet, enqueue it.
                if (isValid(row, col, mazeHeight,mazeWidth) && (grid[row][col] != ('!') && !visitedGrid[row][col]))
                {
                    // mark cell as visited and enqueue it
                    visitedGrid[row][col] = true;
                    //adds one to the distance
                    queueNode Adjcell = new queueNode (new Point(row, col), currentCord.dist + 1 );
                    queue.add(Adjcell);
                }
            }
        }

        return -1;
    }

    static boolean isValid(int row, int col, int mazeHeight, int mazeWidth)
    {
        // return true if row number and
        // column number is in range
        return (row >= 0) && (row < mazeWidth) &&  (col >= 0) && (col < mazeHeight);
    }

}
