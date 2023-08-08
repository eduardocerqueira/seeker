#date: 2023-08-08T16:58:26Z
#url: https://api.github.com/gists/a4c00a4a92ad2036e879026731b0e364
#owner: https://api.github.com/users/ldesdunes

# https://realpython.com/python-walrus-operator/

numbers = [2, 8, 0, 1, 1, 9, 7, 7]

description = {
     "length": (num_length := len(numbers)),
     "sum": (num_sum := sum(numbers)),
     "mean": num_sum / num_length,
}

>>> description
{'length': 8, 'sum': 35, 'mean': 4.375}