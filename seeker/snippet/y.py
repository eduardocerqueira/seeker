#date: 2023-08-15T16:52:25Z
#url: https://api.github.com/gists/abb3988fb1c797a7df876b3936015e5e
#owner: https://api.github.com/users/ChronosX88

Y = lambda f: (lambda x: x(x))(lambda x: f(lambda y: x(x)(y)))

def almost_factorial():
    return lambda f: lambda x: 1 if x == 1 else x * f(x-1) 

def factorial(x):
    return Y(almost_factorial())(x)

print(factorial(6))
