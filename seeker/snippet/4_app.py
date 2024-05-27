#date: 2024-05-27T16:59:45Z
#url: https://api.github.com/gists/e48ba06b8a65d53cbb857a3f323a8b98
#owner: https://api.github.com/users/theSamyak

def apply_operation(operation, x, y):
    return operation(x, y)

# Functions to pass as arguments
def add(x, y):
    return x + y

def multiply(x, y):
    return x * y

# Using the higher-order function
result_add = apply_operation(add, 3, 4)
result_multiply = apply_operation(multiply, 3, 4)

print(result_add)       # Output: 7
print(result_multiply)  # Output: 12