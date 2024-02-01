#date: 2024-02-01T16:47:33Z
#url: https://api.github.com/gists/db7f5ba2c860d2e970785a29abe1d84f
#owner: https://api.github.com/users/mighty-odewumi

def min6(*args):
    if len(args) == 0:
        return None
    return min(args)

# Test the function
result = min6(10, 5, 8, 12, 3, 6)
print("Smallest number:", result)
