#date: 2024-03-28T17:02:47Z
#url: https://api.github.com/gists/bfa5b1fb6d14050360e1ccae3b010e20
#owner: https://api.github.com/users/christian-oudard

def gcd_extended(a, b):
    """
    Compute the greatest common divisor of a and b, along with Bezout
    coefficients s and t, such that gcd(a, b) a*s + b*t.
    Returns (gcd(a, b), s, t, lcm(a, b)).

    23 * 240 - 120 * 46 = 2
    >>> gcd_extended(240, 46)
    (2, -9, 47, 5520)
    """
    if a < b:
        a, b = b, a
    assert a >= b

    r_prev, r_curr = a, b
    s_prev, s_curr = 1, 0
    t_prev, t_curr = 0, 1

    while r_curr > 0:
        q, r_next = divmod(r_prev, r_curr)
        s_next = s_prev - q * s_curr
        t_next = t_prev - q * t_curr

        r_prev, r_curr = r_curr, r_next
        s_prev, s_curr = s_curr, s_next
        t_prev, t_curr = t_curr, t_next
        # assert r_curr == (a*s_curr + b*t_curr)

    lcm = abs(s_curr) * a

    r, s, t = r_prev, s_prev, t_prev
    return (r, s, t, lcm)


def modular_inverse(a, m):
    """
    Compute the modular inverse of a modulo m.

    >>> modular_inverse(4, 18) is None
    True
    >>> modular_inverse(3, 11)
    4
    >>> modular_inverse(
    ...    71252644565283390793,
    ...    4570592454820781536526352207649991860968
    ... )
    64146285133783273633
    """
    gcd, _, t, _ = gcd_extended(m, a)
    if gcd != 1:
        return None
    inv = t % m
    # assert (a * inv) % m == 1
    return inv
