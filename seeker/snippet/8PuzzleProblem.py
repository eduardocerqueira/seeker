#date: 2026-02-26T17:34:36Z
#url: https://api.github.com/gists/d1cf00ddaba0e17d587db8097bb3e84e
#owner: https://api.github.com/users/KrishnaSrinivas-24

import heapq
def print_state(s):
    for r in s:
        print(r)
    print()
def h(s,g):
    d=0
    for i in range(3):
        for j in range(3):
            v=s[i][j]
            if v!=0:
                for x in range(3):
                    for y in range(3):
                        if g[x][y]==v:
                            d+=abs(i-x)+abs(j-y)
    return d
def solve(start,goal):
    pq=[(h(start,goal),start,0,0)]
    visited=set()
    while pq:
        f,state,step,path=heapq.heappop(pq)
        print_state(state)
        print("Heuristic Value:",h(state,goal))
        print("Step Cost:",step)
        print("Path Cost:",path)
        print()
        if state==goal:
            print("Goal Reached")
            return
        visited.add(tuple(map(tuple,state)))
        for i in range(3):
            for j in range(3):
                if state[i][j]==0:
                    moves=[(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
                    for x,y in moves:
                        if 0<=x<3 and 0<=y<3:
                            new=[row[:] for row in state]
                            new[i][j],new[x][y]=new[x][y],new[i][j]
                            if tuple(map(tuple,new)) not in visited:
                                heapq.heappush(pq,
                                (path+1+h(new,goal),
                                new,
                                1,
                                path+1))
print("Enter Initial State:")
start=[list(map(int,input().split())) for i in range(3)]
print("Enter Goal State:")
goal=[list(map(int,input().split())) for i in range(3)]
solve(start,goal)