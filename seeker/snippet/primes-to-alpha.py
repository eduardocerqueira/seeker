#date: 2026-01-21T17:46:09Z
#url: https://api.github.com/gists/9c31c88e005060fabf4695062c0c75d4
#owner: https://api.github.com/users/peteristhegreat

primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
alpha = 'abcdefghijklmnopqrstuvwxyz'

for p,a in zip(primes, alpha):
    print(a, p)

p2a = { p:a for p,a in zip(primes, alpha)}
a2p = { a:p for p,a in zip(primes, alpha)}
def word_to_num(word):
    prod = 1
    for a in word:
        prod *= a2p[a]
    return prod

word_to_num('power')
# 138731263
word_to_num('math')
# 110618
word_to_num('is')
# 1541
138731263/11
# 12611933.0
p2a[11]
# 'e'
def num_to_word(num):
    # try dividing by each prime
    # if it is divisible, add it to the list of prime factorizations
    primes = []
    for p in p2a.keys():
       if num%p == 0:
           primes.append(p)
    for p in primes:
       print(p2a[p], end='')

num_to_word(138731263)