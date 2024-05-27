#date: 2024-05-27T16:59:45Z
#url: https://api.github.com/gists/e48ba06b8a65d53cbb857a3f323a8b98
#owner: https://api.github.com/users/theSamyak

def double(n):
    return n * 2

def map_function(func, values):
    result = []
    for value in values:
        result.append(func(value))
    return result

# Use the custom map function
doubled_values = map_function(double, [3, 6, 9, 12, 15])
print(doubled_values)  # Output: [6, 12, 18, 24, 30]