#date: 2024-05-27T16:59:45Z
#url: https://api.github.com/gists/e48ba06b8a65d53cbb857a3f323a8b98
#owner: https://api.github.com/users/theSamyak

def create_multiplier(n):
    def multiplier(x):
        return x * n
    return multiplier

# Creating specific multiplier functions
multiply_by_2 = create_multiplier(2)
multiply_by_3 = create_multiplier(3)

# Using the returned functions
result1 = multiply_by_2(5)   # Output: 10 (5 * 2)
result2 = multiply_by_3(4)   # Output: 12 (4 * 3)

print(result1)
print(result2)