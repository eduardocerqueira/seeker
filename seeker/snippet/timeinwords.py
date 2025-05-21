#date: 2025-05-21T16:37:37Z
#url: https://api.github.com/gists/4716a4d39c21348545c63ef23df647e0
#owner: https://api.github.com/users/jac18281828


class TimeInWordsError(Exception):
    pass


def numberToString(d: int) -> str:
    """Number to string"""
    number_look = {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
        21: "twenty one",
        22: "twenty two",
        23: "twenty three",
        24: "twenty four",
        25: "twenty five",
        26: "twenty six",
        27: "twenty seven",
        28: "twenty eight",
        29: "twenty nine",
        30: "thirty",
    }
    if d in number_look:
        return number_look[d]
    raise TimeInWordsError("Not a time value")


def timeInWords(h: int, m: int) -> str:
    """
    convert time to a time string representation
    """
    hour_str = numberToString(h)
    next_hour_str = numberToString((h + 1) % 12)
    if m == 0:
        return f"{hour_str} o' clock"
    elif m <= 30:
        if m == 30:
            return f"half past {hour_str}"
        elif m == 20:
            return f"twenty past {hour_str}"
        elif m == 15:
            return f"quarter past {hour_str}"
        elif m == 1:
            return f"one minute past {hour_str}"
        else:
            return f"{numberToString(m)} minutes past {hour_str}"
    else:
        time_until = 60 - m
        if m == 45:
            return f"quarter to {next_hour_str}"
        elif m == 40:
            return f"twenty to {next_hour_str}"
        else:
            return f"{numberToString(time_until)} minutes to {next_hour_str}"


if __name__ == "__main__":
    examples = [(5, 0), (5, 1), (5, 20), (5, 28), (5, 40), (5, 47), (12, 00)]
    for e in examples:
        print(timeInWords(e[0], e[1]))
