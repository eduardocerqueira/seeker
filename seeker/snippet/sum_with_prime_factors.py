#date: 2021-08-31T13:21:10Z
#url: https://api.github.com/gists/08be292422bc0e552e2cffed571b0773
#owner: https://api.github.com/users/steveleec

from find_list_prime_factors import list_prime_factors


def sum_wth_prime_factors(I):
    # We are collecting all factor prime numbers from the array I.
    # Since we do not need to repeat the same values, we use a set.
    # Notice that we are passing and array into 'set'. Because of that,
    # we'll use '.update()' instead of '.add()'. '.update()' receive arrays.
    all_prime_factors = set()
    for n in I:
        if not n in all_prime_factors:
            # collecting all prime factors from the array I
            all_prime_factors.update(list_prime_factors(n))

    result = []
    for prime in all_prime_factors:
        # Find the sum of the elements of array I that have 'prime' as factor
        # To understand this part, know that with Python you can create
        # arrays in this way: Let's see what's happening here:
        # I.    'for val in I' iterates over all elements of array 'I'
        # II.   Since there is a conditional 'if val % prime == 0', it will select
        #       only those elements that meet that criteria
        # III.  The 'val' on the left of 'for val in I...' is what you will store in the array
        vals_with_prime_factor = [val for val in I if val % prime == 0]
        # IV.   The keyword 'sum()' allows you to sum a complete array
        sum_that_have_prime_factor = sum(vals_with_prime_factor)
        # V.    '[prime, sum]' is the format to return
        prime_and_sum = [prime, sum_that_have_prime_factor]
        # VI.   We append the previous array in the final array result
        result.append(prime_and_sum)
        # Steps from I to VI could be done in one like this:
    #       # result.append([prime, sum(val for val in I if val % prime == 0)])

    # VII. The challenge ask us to return a sorted array. 'sort()' help us with that
    #      It will sort keeping in mind the first element of array of two elements
    result.sort()
    return result