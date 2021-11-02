#date: 2021-11-02T17:11:23Z
#url: https://api.github.com/gists/9ab5b2d2d20d49f53bd6751f1559c295
#owner: https://api.github.com/users/GeminiCCCC

class DSU:
    def __init__(self, m, n):
        self.parents = list(range(m*n+1))
        self.ranks = [0] * (m*n+1)
        # keep each components size
        self.size = [1] * (m*n+1)
        # root of 0 means all the bricks connect to the root
        self.size[0] = 0
        
    def find(self, x):
        if x != self.parents[x]:
            self.parents[x] = self.find(self.parents[x])
        
        return self.parents[x]
        
    def union(self, x, y):
        r1 = self.find(x)
        r2 = self.find(y)
        
        if r1 != r2:
            if self.ranks[r1] > self.ranks[r2]:
                self.parents[r2] = r1
                self.size[r1] += self.size[r2]
            elif self.ranks[r1] < self.ranks[r2]:
                self.parents[r1] = r2
                self.size[r2] += self.size[r1]
            else:
                self.parents[r2] = r1
                self.ranks[r1] += 1
                self.size[r1] += self.size[r2]
        

class Solution:
    def hitBricks(self, grid: List[List[int]], hits: List[List[int]]) -> List[int]:
        def unionAround(i, j):
            seq = i * n + j + 1
            
            for d in directions:
                ni = i + d[0]
                nj = j + d[1]
                
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                    new_seq = ni * n + nj + 1
                    dsu.union(seq, new_seq)
            
            # every time sees a brick connects to root, union it with 0, so that all the bricks connect to root will be unioned together
            if i == 0:
                dsu.union(0, seq)
            
        
        m = len(grid)
        n = len(grid[0])
        
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        
        dsu = DSU(m,n)
        
        # remove hits
        for x, y in hits:
            if grid[x][y] == 1:
                grid[x][y] = 2
        
        # union each component
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    unionAround(i, j)
        
        # get cur bricks
        curBricksConnectToRoof = dsu.size[dsu.find(0)]
        hitCount = len(hits)
        
        ans = [0] * hitCount
        
        # adding hit back in reverse order
        for i in range(hitCount-1, -1, -1):
            x = hits[i][0]
            y = hits[i][1]
            
            if grid[x][y] == 2:
                # add hit back
                grid[x][y] = 1
                # union two components got separated by this hit
                unionAround(x,y)
                # get updated bricks count
                newBricksConnectToRoot = dsu.size[dsu.find(0)]
                
                # difference could be 0, and we only update when there is difference
                if newBricksConnectToRoot > curBricksConnectToRoof:
                    # -1 is to remove the hit itself
                    ans[i] = newBricksConnectToRoot - curBricksConnectToRoof - 1
                    curBricksConnectToRoof = newBricksConnectToRoot
        
        return ans
        