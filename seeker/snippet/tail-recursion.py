#date: 2024-09-09T16:59:27Z
#url: https://api.github.com/gists/d714841ebe6da825163b82e31ace9117
#owner: https://api.github.com/users/docsallover

def factorial_tail(n, acc=1):
    if n == 0:
        return acc
    else:
        return factorial_tail(n - 1, n * acc)