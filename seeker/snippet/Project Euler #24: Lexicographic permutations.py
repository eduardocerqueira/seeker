#date: 2025-05-12T17:04:55Z
#url: https://api.github.com/gists/f69e4cb419c772feed48365bdb431740
#owner: https://api.github.com/users/ssokolowskisebastian


import time


start_time = time.time()


def factorial(n):
    if n == 1 or n == 0:
        return 1
    return n*factorial(n-1)


def solution(res):
    result = []
    numbers = [x for x in range(10)]
    while res > 1:
        for j in range(9, -1, -1):
            for i in range(len(numbers)):
                if res > factorial(j):
                    res -= factorial(j)
                else:
                    result.append(numbers[i])
                    numbers.remove(result[-1])
                    break
    return ''.join(map(str, result))


print(solution(1_000_000))


print("--- %s seconds ---" % (time.time() - start_time))
