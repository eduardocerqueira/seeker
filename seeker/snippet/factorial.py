#date: 2024-07-02T16:37:30Z
#url: https://api.github.com/gists/ab0b40e2be853987106a76ceb4352d52
#owner: https://api.github.com/users/gmotzespina

def factorial(num):
    # Base case
    if num == 0:
        return 1
    # Recursive step
    else:
        return num * factorial(num - 1)

print(factorial(3))  # Output: 6