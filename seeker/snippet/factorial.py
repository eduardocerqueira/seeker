#date: 2024-06-25T16:48:23Z
#url: https://api.github.com/gists/9168a7c098b29983b954f1abd33303ae
#owner: https://api.github.com/users/DiscoDancer

def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)

print(factorial(5))