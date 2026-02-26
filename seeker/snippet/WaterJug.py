#date: 2026-02-26T17:34:36Z
#url: https://api.github.com/gists/d1cf00ddaba0e17d587db8097bb3e84e
#owner: https://api.github.com/users/KrishnaSrinivas-24

from collections import deque
def water_jug(a_cap, b_cap, target):
    q = deque([(0,0)])
    visited = set([(0,0)])
    parent = {(0,0): None}
    while q:
        a,b = q.popleft()
        if a == target:
            path=[]
            state=(a,b)
            while state is not None:
                path.append(state)
                state=parent[state]
            path.reverse()
            print("Solution Path:")
            for s in path:
                print(s)
            return
        moves = [
        (0,b),              # Empty A
        (a,0),              # Empty B
        (a_cap,b),          # Fill A
        (a,b_cap),          # Fill B
        (a-min(a,b_cap-b), b+min(a,b_cap-b)),   # A→B
        (a+min(b,a_cap-a), b-min(b,a_cap-a))    # B→A
        ]
        for m in moves:
            if m not in visited:
                visited.add(m)
                parent[m]=(a,b)
                q.append(m)
a_cap=int(input("Enter Jug A Capacity: "))
b_cap=int(input("Enter Jug B Capacity: "))
target=int(input("Enter Target in Jug A: "))
water_jug(a_cap,b_cap,target)