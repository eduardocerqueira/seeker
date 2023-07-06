#date: 2023-07-06T16:58:00Z
#url: https://api.github.com/gists/40601ae31a834064ebe27d86e4e7bdb9
#owner: https://api.github.com/users/WALUNJ1710

def print_arguments(*args, **kwargs):

    for arg in args:

        print(arg)

    for key, value in kwargs.items():

        print(key + ": " + str(value))



print_arguments("Hello", "World", name="John", age=25)

# Output:

# Hello

# World

# name: John

# age: 25