#date: 2024-02-01T16:45:56Z
#url: https://api.github.com/gists/bf1c4977b23b147617faa203b1938af1
#owner: https://api.github.com/users/mighty-odewumi

def square_each(items):
    squared_items = [item ** 2 for item in items]
    return squared_items

# Test the function
input_list = [int(x) for x in input("Enter a list of items separated by space: ").split()]
squared_result = square_each(input_list)
print("Squared items:", squared_result)
