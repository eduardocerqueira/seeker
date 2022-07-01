#date: 2022-07-01T17:16:40Z
#url: https://api.github.com/gists/ab365510f1c5c444e7f993edd78eb14a
#owner: https://api.github.com/users/chinchalinchin

import typing
import random 


def recursive_sum_of_squares(x: List[float], checked: bool = False):
    n = len(x)

    if not checked:
        if not all(this_x is not None and isinstance(this_x, (float, int)) for this_x in x):
            raise ValueError(
                'Sample contains null values')

        if n == 0:
            raise ValueError(
                'Sample variance cannot be computed for a sample size of 0.')

    if n == 1:
        return 0

    term_variance = (n*x[-1] - sum(x))**2/(n*(n-1))
    return recursive_sum_of_squares(x[:-1], True) + term_variance

if __name__ == "__main__":
    sample = [random.randint(0, 100) for _ in range(0,100)]
    n = len(sample)
    variance = recursive_sum_of_squares(sample) / (n - 1)
    print(variance)