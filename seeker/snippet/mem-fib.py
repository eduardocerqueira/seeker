#date: 2023-11-23T16:34:06Z
#url: https://api.github.com/gists/a98db074d92f5dd5b3256166a3991450
#owner: https://api.github.com/users/josh-dev-things

import time

memory = {}
def memFib(n:int): 
    seen = n in memory
    if(seen):
        return memory[n]

    if(n <= 0):
        return 0
    
    if(n <= 2):
        return 1
    
    fibVal = memFib(n-1) + memFib(n-2)

    #Write to memory
    if(not seen):
        memory[n] = fibVal

    return fibVal

n1 = 20
n2 = 21

t1 = time.time()
memFib(n1)
t2 = time.time()
memFib(n2)
t3 = time.time()

print("Time for execution for: " + str(n1) + " was: " + str(t2-t1) + "s")
print("Time for exectution for: " + str(n2) + " was: " + str(t3-t2) + "s")

# The point of this program is to show the memoisation optimisation for recursive algorithms