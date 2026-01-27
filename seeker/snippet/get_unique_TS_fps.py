#date: 2026-01-27T17:21:31Z
#url: https://api.github.com/gists/3cb51a0bb498326b2df0b5b9df11f9f8
#owner: https://api.github.com/users/dehaenw

from collections import Counter

def get_unique_TS_fps(N):
    onbits = [i+1 for i in range(N)][::-1]
    for _ in range(N):
        sims = set([])
        problem_idx = []
        for i,ob1 in enumerate(onbits):
            for j,ob2 in enumerate(onbits):
                if i>j: #guarantees ob1 is larger
                    sim = ob2/ob1
                    if sim in sims:
                        problem_idx += [i,j]
                    else:
                        sims.add(sim)
        mc = Counter(problem_idx).most_common(1)
        if len(mc)>0:
            onbits.pop(mc[0][0])
        else:
            break
    return ["1"*nbits+"0"*(N-nbits) for nbits in onbits]      
fps = get_unique_TS_fps(1024)
for fp in fps:
    print(fp)