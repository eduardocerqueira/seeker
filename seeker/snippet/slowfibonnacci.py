#date: 2025-11-12T17:13:03Z
#url: https://api.github.com/gists/aa7145ec23946f5cd38967677c0670a8
#owner: https://api.github.com/users/RebornGhost

def slow_fibonacci(n):
    if n == 1 or n == 2:
        return 1
    else:
        return slow_fibonacci(n - 1) + slow_fibonacci(n - 2)

for n in range(1, 31):  # 30 is already quite slow!
    print(n, ":", slow_fibonacci(n))
