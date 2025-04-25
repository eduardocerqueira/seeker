#date: 2025-04-25T16:36:27Z
#url: https://api.github.com/gists/35bc9780f0745b7dd5893d883da2d356
#owner: https://api.github.com/users/ssokolowskisebastian

import time

start_time = time.time()



def solution():
    res = 0
    p = 0
    for p in range(12,1000):
        solutions = 0
        for a in range(1, p//3):
            for b in range(a,p-2*a):
                c = (b**2 + a**2) ** 0.5
                if c%1 == 0 and a+b+c == p:
                    solutions +=1
        if solutions > res:
            res = solutions
            p_value = p
           
    return (res,p_value)
            

print(solution())

print("--- %s seconds ---" % (time.time() - start_time))  