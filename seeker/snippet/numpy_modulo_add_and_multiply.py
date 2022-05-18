#date: 2022-05-18T16:59:39Z
#url: https://api.github.com/gists/64c15e63c92a48e4e4a3132f889499b8
#owner: https://api.github.com/users/soldni

import numpy as np
from functools import lru_cache

@lru_cache()
def get_tmax(t: str) -> np.ndarray:
    return np.array(np.iinfo(t).max, dtype=t)


@lru_cache()
def get_tmin(t: str) -> np.ndarray:
    return np.array(np.iinfo(t).min, dtype=t)


@lru_cache()
def get_half_max(t: str) -> np.ndarray:
    return get_tmax(t) >> 1


@lru_cache()
def get_half_min(t: str) -> np.ndarray:
    return get_tmin(t) >> 1


@lru_cache()
def zero(t: str) -> np.ndarray:
    return np.array(0, dtype=t)


@lru_cache
def one(t: str) -> np.ndarray:
    return np.array(1, dtype=t)


def check_if_overlow(*terms, op='__add__'):
    """Returns True if this operation will result in overflow,
    False otherwise"""
    acc, *terms = terms
    half_min = get_half_min()
    half_max = get_half_max()
    acc = acc >> 1
    for t in terms:
        acc = getattr(acc, op)(t >> 1)
        if not half_min <= acc <= half_max:
            return True
    return False


def modulo_add(a: np.ndarray,
               b: np.ndarray,
               mod: np.ndarray,
               _offset: np.ndarray = None):
    """Sum `a` and `b` modulo `mod`. If `_offset` is provided,
    it is used to offset the calculation when appropriate.
    Offset is typically not provided when calling the function,
    but it internally used when `mod` is negative.
    """

    # Setting the offset to 0 if it is not provided
    _offset = _offset or zero(mod.dtype)

    # It makes no sense to add two numbers that are greater
    # than the modulo; we rescale first here.
    a %= mod
    b %= mod

    # Shortcut if a or b are 0: the result of the addition is
    # the other number plus any offset.
    if not a:
        return b + _offset
    if not b:
        return a + _offset

    # For cases when the modulo is negative, we reduce it to
    # the case of positive modulo.
    if mod < 0:

        # Particular care needs to be put when flipping the
        # sign of the modulo. If the modulo is at the edge of
        # the allowed range of values for this type (e.g. -128
        # for int8), then mod == -mod, which would not be bueno.
        # So if we are at the edge, we shift the value by one
        # (e.g. to -127) and then flip the sign. This requires
        # us a bit of finessing ong the final result, which is why
        # we need an offset.
        if mod == -mod:
            flipped_mod = -(mod - np.sign(mod))
            offset = one(mod.dtype)
        else:
            flipped_mod = -mod
            offset = zero(mod.dtype)

        # We perform the actual sum here.
        result = modulo_add(a, b, flipped_mod, _offset=offset)

        # If the result is positive (i.e., result + offset > 0),
        # we shift the result by the modulo.
        if result > -offset:
            # We need to do this sum in a very specific order,
            # as adding `offset` to `result` first might cause
            # an overflow.
            return result + mod + offset
        else:
            return result + offset

    # Two cases here; code is mostly from this answer on SO
    # https://stackoverflow.com/a/11249135 , but adapted to cover
    # a couple of corner cases due to negative integers.
    if (
        # if subtracting m from b causes an overflow, we don't
        # do that!
        not check_if_overlow(mod, -b)
        and a >= (mod - b)
    ):
        result = a - mod + b
        return result
    else:
        result = b + a + _offset
        result += mod if result <= 0 else np.zeros_like(mod)
        return result


def modulo_multiply(a: np.ndarray,
                    b: np.ndarray,
                    mod: np.ndarray) -> np.ndarray:
    """Multiplies `a` and `b` modulo `mod`. If `b` is negative,
    it calculates a * b modulo `mod` first, and then it shifts
    the results appropriately.
    """
    result_accumulator = zero(mod.dtype)

    # We first scale a by mod, as this does not change the
    # final result.
    a %= mod

    # Shortcut in case b is a multiple of mod: the result is 0
    if b % mod == 0:
        return result_accumulator
    
    # Code from now on is derived from 
    # geeksforgeeks.org/multiply-large-integers-under-large-modulo/
    # but with tweaks to handle negative numbers. 
    
    # If b is negative, we multiply by -b first, and then shift
    # the result by `a` times modulo `mod`.
    if b < 0:
        if b == -b:
            # number range is not symmetrical, so if b=-128,
            # negating b gets us to -128 again (assuming int8).
            # Therefore, we step off by one by using -(sign(b)).
            b_ = -(b - np.sign(b))

            # If we are in the special case of b == -b, we
            # need to do a trick here similar to the offset
            # we used when adding. When adding, sometimes results
            # are off by 1/2 if a factor/modulo value are at
            # the boundary of allowed values. It's the same for
            # multiplications, except that if b is at the boundary,
            # we are off by a factor of a * 1. Therefore, we add
            # 1 a-times to r modulo r. We set the flag here, but
            # do rotation later.
            offset = True
        else:
            b_ = -b
            offset = False

        # We perform the actual multiplication here.
        result = modulo_multiply(a, b_, mod)
        if offset:
            # Apply the offset in case we are working with lower bound
            # of integer range.
            result = modulo_add(result, a, mod)

        # 0 is a fixed point, so no need to shift the result in that base
        return mod - result if result != 0 else result

    while b > 0:

        if b & 1:
            # If b is odd, we add `a` once and then shift digits of the right.
            result_accumulator = modulo_add(result_accumulator, a, mod)

        # We double before shifting for faster computation.
        a = modulo_add(a, a, mod)

        b >>= 1  # this is the same as dividing by 2, but faster.

    return result_accumulator