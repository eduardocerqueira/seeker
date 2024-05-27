#date: 2024-05-27T16:59:45Z
#url: https://api.github.com/gists/e48ba06b8a65d53cbb857a3f323a8b98
#owner: https://api.github.com/users/theSamyak

def outer_scope():
    name = 'Samyak'
    city = 'New York'

    def inner_scope():
        print(f"Hello {name}, Greetings from {city}")

    return inner_scope
    
# Assigning the inner function to a variable
greeting_func = outer_scope()

# Calling the inner function
greeting_func()