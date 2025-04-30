#date: 2025-04-30T17:02:59Z
#url: https://api.github.com/gists/c713f9c695f3870141ef1e481707683e
#owner: https://api.github.com/users/ssokolowskisebastian

import time

start_time = time.time()



def get_primes(limit):
    for i in range(2,limit+1):
        for j in range(2,int(i**0.5)+1):
            if i%j == 0:
                break
        else:
            yield i
    
def is_prime(n):
    for i in range(2,int(n**0.5)+1):
        if n%i == 0:
            return False
    return True


def solution(limit):
    
    max_result = 0
    max_count = 0
    primes = list(get_primes(limit))
    n = len(primes)
    for i in range(n):
        temp_sum= primes[i]
        count = 1
        for j in range(i+1,n):
            temp_sum+= primes[j]
            count+=1
            
            if temp_sum>1_000_000 :
                break
            
            if is_prime(temp_sum) :
                if count > max_count:
                    max_count = count
                    max_result = temp_sum

                
    return max_result, max_count

print(solution(4000))
    

print("--- %s seconds ---" % (time.time() - start_time))   