#date: 2023-07-06T16:59:03Z
#url: https://api.github.com/gists/5abaa0b32233e8223d8dedb0c3ac75f6
#owner: https://api.github.com/users/WALUNJ1710

def factorial(n):

    if n == 0 or n == 1:

        return 1

    else:

        return n * factorial(n - 1)



result = factorial(5)

print(result)  # Output: 120