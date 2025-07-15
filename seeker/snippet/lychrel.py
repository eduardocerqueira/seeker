#date: 2025-07-15T16:54:38Z
#url: https://api.github.com/gists/4a9ad76636e7d5388810655abc02e636
#owner: https://api.github.com/users/Pearson69

def is_palindrome(n):
    return str(n) == str(n)[::-1]

def is_lychrel(n, iterations=200):
    for _ in range(iterations):
        n += int(str(n)[::-1])
        if is_palindrome(n):
            return False
    return True


for i in range(1, 10000):
    if is_lychrel(i):
        print(f"{i} is a Lychrel number.")
