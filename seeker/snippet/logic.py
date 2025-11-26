#date: 2025-11-26T17:01:31Z
#url: https://api.github.com/gists/717e7307bc2e96f6d90c9918c89c35be
#owner: https://api.github.com/users/themysterysolver

import re
from collections import deque
def algoSolver(at,bt,option,tq=None,p=None):
    at = list(map(int,re.split(r'\s+',at.strip())))
    bt = list(map(int,(re.split(r'\s+',bt.strip()))))
    if len(at)!=len(bt):
        return None
    if p:
        priority = list(map(int,(re.split(r'\s+',p.strip()))))
        if len(priority)!=len(at):
            return False
    if tq:
        tq = int(tq)
    # print(option)
    # print(tq,p)
    ans  = {}
    if option == "First come first served(FCFS)":
        ans = FCFS(at,bt)
    elif option == "Shortest job first(SJF)":
        ans = SJF(at,bt)
    elif option == "Shortest remaining time first(SRTF)":
        ans = SRTF(at,bt)
    elif option == "Premptive Priority scheduling":
        ans = PPS(at,bt,priority)
    elif option == "Non-Premptive priority scheduling":
        ans =  NPPS(at,bt,priority)
    elif option == "Round robin(RR)":
        ans = RR(at,bt,tq)

    return ans

def FCFS(at,bt):
    n = len(at)
    ct = [0]*n
    tat = [0]*n
    wt = [0]*n

    process = sorted(range(n),key=lambda i:at[i])

    time = 0
    for i in process:
        if time<at[i]:
            time = at[i]
        time+=bt[i]
        ct[i] = time
        tat[i] = ct[i]-at[i]
        wt[i] = tat[i]-bt[i]

    return {"Process":[chr(ord('A')+i) for i in range(n)],"Arrival Time":at,"Burst Time":bt,"Completion Time":ct,"Turn Around Time":tat,"Waiting Time":wt}

def SJF(at,bt):
    n = len(at)
    ct = [0]*n
    tat = [0]*n
    wt = [0]*n
    
    completed = [False]*n
    done = 0
    time = 0

    process = sorted(range(n),key=lambda i:at[i])

    while done<n:
        idx = -1
        minbt = float('inf')
        for i in process:
            if not completed[i] and at[i]<=time:
                if bt[i]<minbt:
                    idx = i
                    minbt = bt[i]

        if idx==-1:
            time+=1
        else:
            time+=bt[idx]
            ct[idx] = time
            completed[idx] = True
            done+=1
            tat[idx] = ct[idx]-at[idx]
            wt[idx] = tat[idx]-bt[idx]


    return {"Process":[chr(ord('A')+i) for i in range(n)],"Arrival Time":at,"Burst Time":bt,"Completion Time":ct,"Turn Around Time":tat,"Waiting Time":wt}        
    
def NPPS(at,bt,priority):
    n = len(at)
    ct = [0]*n
    tat = [0]*n
    wt = [0]*n

    process = sorted(range(n),key=lambda i:at[i])

    time = 0
    completed = [False]*n
    finished = 0

    while finished<n:
        idx = -1
        prio = float('inf')
        for i in process:
            if not completed[i] and at[i]<=time:
                if priority[i]<prio:
                    prio = priority[i]
                    idx = i
        
        if idx == -1:
            time+=1
        else:
            time+=bt[idx]
            ct[idx] = time
            completed[idx] = True
            finished+=1
        
    for i in range(n):
        tat[i] = ct[i] - at[i]
        wt[i] = tat[i] - bt[i]

    return {"Process":[chr(ord('A')+i) for i in range(n)],"Arrival Time":at,"Burst Time":bt,"Completion Time":ct,"Turn Around Time":tat,"Waiting Time":wt}        

def SRTF(at,bt):
    n = len(at)
    ct = [0]*n
    wt = [0]*n
    tat = [0]*n
    completed = [False]*n
    finished = 0
    time = 0

    rem = bt[::] #lazy to chnage that here should have used and decremeneted this all time!

    process = sorted(range(n),key=lambda i:at[i])
    while finished<n:
        idx = -1
        minbt = float('inf')
        for i in process:
            if not completed[i] and at[i]<=time:
                if bt[i]<minbt:
                    minbt = bt[i]
                    idx = i
        if idx == -1:
            time+=1
        else:
            time+=1
            bt[idx]-=1
            if bt[idx] == 0:
                completed[idx] = True
                finished+=1
                ct[idx] = time
    
    for i in process:
        tat[i] = ct[i]-at[i]
        wt[i] = tat[i]-rem[i]
        
    return {"Process":[chr(ord('A')+i) for i in range(n)],"Arrival Time":at,"Burst Time":rem,"Completion Time":ct,"Turn Around Time":tat,"Waiting Time":wt}
    
def PPS(at,bt,priority):
    n = len(at)
    ct = [0]*n
    tat = [0]*n
    wt = [0]*n

    process = sorted(range(n),key=lambda i:at[i])

    rem = bt[::]

    time = 0
    completed = [False]*n
    finished = 0

    while finished<n:
        idx = -1
        prio = float('inf')
        for i in process:
            if not completed[i] and at[i]<=time:
                if priority[i]<prio:
                    prio = priority[i]
                    idx = i
        
        if idx == -1:
            time+=1
        else:
            time+=1
            bt[idx]-=1
            if bt[idx] == 0:
                completed[idx] = True
                finished+=1
                ct[idx] = time
        
    for i in range(n):
        tat[i] = ct[i] - at[i]
        wt[i] = tat[i] - rem[i]
    # print(rem)
    # print(bt)
    return {"Process":[chr(ord('A')+i) for i in range(n)],"Arrival Time":at,"Burst Time":rem,"Completion Time":ct,"Turn Around Time":tat,"Waiting Time":wt}        

def RR(at,bt,tq):
    n = len(at)
    ct = [0]*n
    tat = [0]*n
    wt = [0]*n

    process = sorted(range(n),key=lambda i:at[i])

    rem = bt[::]
    rq = deque([]) #ready queue
    time = 0
    completed = [False]*n
    finished = 0
    gidx = 0
    while finished<n:
        while gidx<n and not completed[gidx] and at[gidx]<=time :
            rq.append(gidx)
            gidx+=1
        
        if not rq:
            next_arrival = min([at[i] for i in range(n) if not completed[i]])
            time = max(time, next_arrival)
            continue
        
        idx = rq.popleft()
        extime = min(tq,rem[idx]) #executing time
        rem[idx]-=extime
        time+=extime

        while gidx<n and not completed[gidx] and at[gidx]<=time :
            rq.append(gidx)
            gidx+=1
        
        if rem[idx]>0:
            rq.append(idx)
        else:
            ct[idx] = time
            finished+=1

    for i in process:
        tat[i] = ct[i]-at[i]
        wt[i] = tat[i]-bt[i]

    return {"Process":[chr(ord('A')+i) for i in range(n)],"Arrival Time":at,"Burst Time":bt,"Completion Time":ct,"Turn Around Time":tat,"Waiting Time":wt}

