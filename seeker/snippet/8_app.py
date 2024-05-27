#date: 2024-05-27T16:59:45Z
#url: https://api.github.com/gists/e48ba06b8a65d53cbb857a3f323a8b98
#owner: https://api.github.com/users/theSamyak

def outer_scope(name, city):

    def inner_scope():
        print(f"Hello {name}, Greetings from {city}")

    return inner_scope

# Creating closures with different names and locations
greet_priyanshu = outer_scope('Dr Priyanshu', 'Jaipur')
greet_sam = outer_scope('Sam', 'New York')

# Executing the closures
greet_priyanshu()    # Output: Hello Dr Priyanshu, Greetings from Jaipur
greet_sam()     # Output: Hello Sam, Greetings from New York