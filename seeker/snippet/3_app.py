#date: 2024-05-27T16:59:45Z
#url: https://api.github.com/gists/e48ba06b8a65d53cbb857a3f323a8b98
#owner: https://api.github.com/users/theSamyak

def create_multiplier(factor):
    """Returns a function that multiplies its input by the given factor."""
    def multiplier(x):
        return x * factor
    return multiplier

# Create specific multiplier functions
double = create_multiplier(2)
triple = create_multiplier(3)

# Use the created functions
print(double(5))  # Output: 10
print(triple(5))  # Output: 15