#date: 2024-02-01T16:46:53Z
#url: https://api.github.com/gists/a8849848b7e8b66e99d5d7a391cbd3ff
#owner: https://api.github.com/users/mighty-odewumi

def swap(a, b):
    return b, a

# Test the function
x = int(input("Enter the first integer: "))
y = int(input("Enter the second integer: "))

print("Before swapping: x =", x, "and y =", y)

x, y = swap(x, y)

print("After swapping: x =", x, "and y =", y)
