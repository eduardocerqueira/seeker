#date: 2021-08-31T13:20:34Z
#url: https://api.github.com/gists/ec28188c9947234b185b275da469b052
#owner: https://api.github.com/users/steveleec

def list_prime_factors(n):
    # n could be negatibe so we make it positive.
    # we don't care the sign, only the prime factors
    if n < 0:
        n = -n

    # We'll use the data structure 'set'.
    # It will help us to get unique values by default.
    # 'set' uses the '.add' property for adding values to its set.
    # It does it in a way that does not repeat values.
    prime_factors = set()

    # If number is a multiple of 2, we only care knowing that its
    # factor is two, other than that we could simplify it.
    while n % 2 == 0:
        prime_factors.add(2)
        n = n / 2

    # Optimization technique: square root of 'n' rather than 'n'.
    # Use 'int()' for converting the value to an integer.
    for i in range(3, int(n ** 0.5) + 1):
        # if 'n' is divisible by 'i', 'i' would be a prime factor
        # that is because we are dividing 'n' by 'i' in a while loop leaving only
        # prime factors
        while n % i == 0:
            prime_factors.add(int(i))
            n = n / i

    # In case 'n = n / i' is not enough for simplifying 'n' until 1.
    # We conclude that that number 'n' must be a prime number factor
    if n > 2:
        prime_factors.add(int(n))

    # so far 'prime_factors' is a set that looks like this:
    # {3, 5, 7, ...}. To convert a set into an array we could
    # use 'sorted' and get the following [3, 5, 7, ...], which
    # will be more useful later on
    return sorted(prime_factors)