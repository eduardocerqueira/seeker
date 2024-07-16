#date: 2024-07-16T17:10:01Z
#url: https://api.github.com/gists/606cf5dded3e5c724c950622dab237db
#owner: https://api.github.com/users/MaXVoLD

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
primes = []
not_primes = []
for i in numbers:
    if i == 1:
        continue
    for j in primes:
        if i % j == 0:
            not_primes.append(i)
            break
    else:
            primes.append(i)
print(f'Простые числа: {primes} \nСоставные числа: {not_primes}')