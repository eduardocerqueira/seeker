#date: 2024-06-06T17:02:26Z
#url: https://api.github.com/gists/ac18d63c3e73048438657728e4ec25c5
#owner: https://api.github.com/users/Leshka60

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
primes = []
not_primes = []
for i in numbers:
    if i == 1:
        continue
    for j in range(2, i):
        if i % j == 0:
            not_primes.append(i)
            break
    else:
        primes.append(i)
print('Primes: ', primes)
print('Not Primes: ', not_primes)
