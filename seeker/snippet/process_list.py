#date: 2025-06-09T16:54:42Z
#url: https://api.github.com/gists/5fffc99b3aecc2cb71a607b66c2ef0e5
#owner: https://api.github.com/users/neocliff

def process_list(lst):
    for item in lst:
        if isinstance(item, list):
            # Recursively process nested list
            process_list(item)
        else:
            # Process primitive type (e.g., print)
            print(item)

# Example usage:
data = [1, 'a', [2, 3, ['b', 4]], 5]
process_list(data)
