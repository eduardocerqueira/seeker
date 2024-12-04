#date: 2024-12-04T16:57:45Z
#url: https://api.github.com/gists/b8e271643dc0108d83daa26388915895
#owner: https://api.github.com/users/ozzloy

def decorator_func(func):
    def manipulated_func(fruit):
        return "manipulated " + func(fruit)
    return manipulated_func

@decorator_func
def return_fruit(fruit):
    return "your fruit is " + fruit

print(return_fruit("pear"))
# manipulated your fruit is pear