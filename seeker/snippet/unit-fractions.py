#date: 2023-11-01T17:05:05Z
#url: https://api.github.com/gists/4dc7eeba8543c11e9b9d77a3d20e4bad
#owner: https://api.github.com/users/Radcliffe

from fractions import Fraction
import sys

"""
Represent a positive rational number as the sum of exactly k unit fractions, distinct or not.
This function returns the first solution in lexicographic order, represented as a list of denominators.
If no solution exists, the function returns None.

We can use the following observation to limit the search space to a finite number of possibilities.
Suppose that r = 1/n_1 + 1/n_2 + ... + 1/n_k, where n_1 ≤ n_2 ≤ ... ≤ n_k.
Then 1/(k * n_1) ≤ r < 1/n_1, which implies that n_1 ≤ floor(r) + 1 and n_1 ≥ floor(r/k).
Thus, there are only a finite number of possible values for n_1.
By recursion, the same is true for n_2, n_3, and so on.

Usage examples:

>  python unit-fractions.py 171523 84882 6
[1, 2, 3, 7, 43, 47]

> python unit-fractions.py 171523 84882 6 0
[1, 1, 49, 3178, 16859688, 378998755750104]
"""


def unit_fraction_rep(frac, num_terms, distinct=1, start=1):
    if frac <= 0:
        return
    if num_terms == 1:
        if frac.numerator == 1:
            return [frac.denominator]
    else:
        start = max(start, int(1 / frac) + 1)
        stop = int(num_terms / frac)
        for denom in range(start, stop + 1):
            new_frac = frac - Fraction(1, denom)
            partial_result = unit_fraction_rep(new_frac, num_terms - 1, distinct, denom + distinct)
            if isinstance(partial_result, list):
                return [denom] + partial_result


if __name__ == '__main__':
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    num_terms = int(sys.argv[3])
    try:
        distinct = int(sys.argv[4])
    except:
        distinct = 1
    rep = unit_fraction_rep(Fraction(a, b), num_terms, distinct)
    print(rep)
