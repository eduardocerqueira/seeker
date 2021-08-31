#date: 2021-08-31T13:19:01Z
#url: https://api.github.com/gists/4d52c980bd8e2dc7c8f48e37a0a6c47b
#owner: https://api.github.com/users/steveleec

def is_prime(n):
    # when n is 1 or less than 1:
    if n <= 1:
        return False

    # range(2, n) will iterate until n - 1
    for div in range(2, n):
        # if the modulus of n % div is 0, it means that
        # n is divisible by a given div, which makes it not a prime number
        if n % div == 0:
            return False

    # True if n was not divisible for the range between 2 to n - 1
    return True
