#date: 2022-02-09T16:51:17Z
#url: https://api.github.com/gists/651a376cad07058c1b7812a27f889a71
#owner: https://api.github.com/users/Comaroff

def sumfibonacci(n):
    if n == 1:
        return 0
    if n == 2:
        return 1
    else:
        return sumfibonacci(n - 2) + sumfibonacci(n - 1)

print(sumfibonacci(8))