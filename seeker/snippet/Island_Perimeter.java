//date: 2022-06-01T17:01:08Z
//url: https://api.github.com/gists/d8758c232f0ff848e0782294d40de547
//owner: https://api.github.com/users/Navila48

class Solution {
    public int islandPerimeter(int[][] grid) {
        int peri=0;
        int r=grid.length;
        int c=grid[0].length;
        boolean[][] visit=new boolean[r][c];
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                if(grid[i][j]==1)
                  peri+= dfs(i,j,grid,peri,visit);
            }
        }
        return peri;        
    }
    public int dfs(int i,int j,int[][]grid,int peri,boolean[][]visit){
        if(i>=grid.length || j>=grid[0].length || i<0 || j<0 || grid[i][j]==0){
            return 1;
        }
        if(visit[i][j])
            return 0;
        visit[i][j]=true;
        peri=dfs(i+1,j,grid,peri,visit);
        peri+=dfs(i,j+1,grid,peri,visit);
        peri+=dfs(i-1,j,grid,peri,visit);
        peri+=dfs(i,j-1,grid,peri,visit);
        return peri;
    }
}