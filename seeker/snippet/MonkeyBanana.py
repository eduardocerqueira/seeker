#date: 2026-02-26T17:34:36Z
#url: https://api.github.com/gists/d1cf00ddaba0e17d587db8097bb3e84e
#owner: https://api.github.com/users/KrishnaSrinivas-24

from collections import deque
LOC = ["entrance","table","window"]
def goal(s):
    return s["banana"]
def next_states(s):
    m,b,ba,on = s["monkey"],s["box"],s["place"],s["onbox"]
    states=[]
    if not on:
        for l in LOC:
            if l!=m:
                ns=s.copy()
                ns["monkey"]=l
                states.append((ns,"Walk to "+l))
    if not on and m==b:
        for l in LOC:
            if l!=b:
                ns=s.copy()
                ns["monkey"]=l
                ns["box"]=l
                states.append((ns,"Push box to "+l))
    if not on and m==b==ba:
        ns=s.copy()
        ns["onbox"]=True
        states.append((ns,"Climb box"))
    if on and m==ba:
        ns=s.copy()
        ns["banana"]=True
        states.append((ns,"Grab banana"))
    return states
def bfs(start):
    q=deque([(start,[])])
    visited=set()
    while q:
        s,path=q.popleft()
        sid=tuple(s.values())
        if sid in visited:
            continue
        visited.add(sid)
        if goal(s):
            return path
        for ns,a in next_states(s):
            q.append((ns,path+[a]))
print("Locations:",LOC)
m=input("Monkey position: ")
b=input("Box position: ")
ba=input("Banana position: ")
start={
"monkey":m,
"box":b,
"place":ba,
"onbox":False,
"banana":False
}
sol=bfs(start)
print("\nSolution:")
for i in range(len(sol)):
    print(i+1,sol[i])