#date: 2021-09-10T16:44:46Z
#url: https://api.github.com/gists/75cf1fafd841ed950225d3348fca1a1f
#owner: https://api.github.com/users/junedsheikh

def factorial_iterative(n):

    fac = 1
    for i in range(n):
        fac = fac * (i+1)
    return fac

n = int(input("Enter number to find factorial = "))
print("Factorial using Iterative = ", factorial_iterative(n))

def factorial_recursive(n):

    if n == 1:
        return 1

    else:
        return n * factorial_recursive(n-1)
print("Factorial using Recursive = ", factorial_recursive(n))