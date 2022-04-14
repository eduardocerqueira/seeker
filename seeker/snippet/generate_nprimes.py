#date: 2022-04-14T17:13:23Z
#url: https://api.github.com/gists/c377e0d2e1dc933e252e776e31f058dc
#owner: https://api.github.com/users/karhunenloeve

def gen_primes(n):
    list_of_primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, n + 1):
        if is_prime[i]:
            list_of_primes.append(i)
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return list_of_primes


def get_amount_of_primes(n):
    return len(gen_primes(n))