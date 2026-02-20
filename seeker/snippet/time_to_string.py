#date: 2026-02-20T17:09:10Z
#url: https://api.github.com/gists/e1c904c74e8771069c606cdb5a6ec7e3
#owner: https://api.github.com/users/marcocamma

import math

_group_to_suffix = ["s", "ms", "us", "ns", "ps", "fs", "as"]


def time_to_str(delay, fmt=4):
    """
    A short function to convert floating numbers (representing numbers from 1e-18 to ~1000) as string, e.g. time_to_str(102.4e-9) to 102.4ns
    Parameters
    ----------
    delay: a float number
    fmt : integer or formatting_string
        if integer, it is used to determine the number of significant digits

    Examples
    --------
    time_to_str(-1.007e-6,fmt=3) → '-1.01us'
    time_to_str(-1.007e-6,fmt=2) → '-1.0us'
    time_to_str(908.2e-6,fmt=2) →   '908us'
    time_to_str(908.2e-6,fmt=1) →   '908us'
    """
    if not isinstance(fmt, (int, str)):
        raise ValueError(
            "The fmt variable must be an integer (number of significant digits) or a formatting string"
        )
    a_delay = abs(delay)
    if a_delay >= 1000:
        return str(delay) + "s"
    group = int(-(math.log10(a_delay * 1.00001)) // 3) + 1
    factor = 10 ** (3 * group)
    if isinstance(fmt, int):
        numer_of_integer_digits = int(math.log10(a_delay * factor * 1.00001))
        n = fmt - numer_of_integer_digits - 1
        if n < 0:
            n = 0
        fmt = f"%+.{n}f"
    ret = fmt % (delay * factor) + _group_to_suffix[group]
    return ret