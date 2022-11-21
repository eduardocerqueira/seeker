#date: 2022-11-21T17:07:06Z
#url: https://api.github.com/gists/4328ac5d67f89a7b1d74b25c9cf8eb63
#owner: https://api.github.com/users/arikchakma

n = int(input("Enter any Real number: "))

for i in range(n):
    for j in range(i + 1):
        print("*", end=" ")
    print()

# Result:
# *
# * *
# * * *
# * * * *
# * * * * *

for i in range(n):
    for j in range(i, n):
        print("*", end=" ")
    print()

# Result:
# * * * * *
# * * * *
# * * *
# * *
# *

for i in range(n):
    for j in range(i + 1):
        print(" ", end=" ")
    for j in range(i, n):
        print("*", end=" ")
    print()

# Result:
#   * * * * *
#     * * * *
#       * * *
#         * *
#           *

for i in range(n):
    for j in range(i + 1):
        print(" ", end=" ")
    for j in range(i, (n - 1)):
        print("*", end=" ")
    for j in range(i, n):
        print("*", end=" ")
    print()

# Result:
#   * * * * * * * * *
#     * * * * * * *
#       * * * * *
#         * * *
#           *
