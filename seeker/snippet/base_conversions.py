#date: 2024-09-16T17:00:30Z
#url: https://api.github.com/gists/dd83ecd985ad1817d72ae92764b4921c
#owner: https://api.github.com/users/Marcus5408

# conversion.py
# -------------
# Description:
# A simple program that converts a number from one base to another using either
# the successive division method or the weighted multiplication method.
# -------------
# Usage:
# In a terminal, run the following command:
# python3 conversion.py <method> <number> <base>
#
# <method>  selects the conversion method using one of the following:
#           - divide: successive division method
#           - multiply: weighted multiplication method
# <number>  is the number to convert.
# <base>    is the target base (for successive division)
#           or the base of the number (for weighted multiplication).
# -------------
# (c) Issac Liu, 2024

from typing import Union, Literal
import sys


def success_div(n, base):
    remainder = 0
    result = 0
    charset = "0123456789"
    if base > 10:
        if base == 16:
            charset = "0123456789ABCDEF"
        else:
            print(
                "You have entered a base greater than 10. Please enter every digit of your base from least to greatest."
            )
            values = input("")
            charset = values if len(values) == base else "0123456789ABCDEF"
    if base < 10:
        while n != 0 or n > base:
            remainder = n % base
            quotient = n // base
            print(f"{n}/{base} = {quotient}r{remainder}")
            result = result * 10 + remainder
            n = quotient
        # reverse the result
        result = int(str(result)[::-1])
        print(f"\n{result}")
    else:
        result = ""
        while n != 0:
            remainder = n % base
            quotient = n // base
            if base > 10 and remainder > 9:
                hex_value = f" ({remainder} -> {charset[remainder]})"
                print(f"{n}/{base} = {quotient}r{remainder}{hex_value}")
            else:
                print(f"{n}/{base} = {quotient}r{remainder}")
            result = charset[remainder] + result
            n = quotient
        print(f"\n{result}")

    return result


def weighted_multiply(n: Union[int, str], base: int) -> int:
    if isinstance(n, str):
        n = n.upper()
        charset = "0123456789ABCDEF"
        list = [charset.index(x) for x in n]
    else:
        list = [int(x) for x in str(n)]

    weights = [base**i for i in range(len(list) - 1, -1, -1)]
    result = [a * b for a, b in zip(list, weights)]

    for i in range(len(result)):
        if base > 10 and list[i] > 9:
            hex_value = f" ({charset[list[i]]} -> {list[i]})"
            print(
                f"{list[i]}{hex_value} * {base}^{len(list) - i - 1} = {list[i]} * {weights[i]} = {result[i]}"
            )
        else:
            print(
                f"{list[i]} * {base}^{len(list) - i - 1} = {list[i]} * {weights[i]} = {result[i]}"
            )

    print(f"\n{' + '.join([str(x) for x in result])} = {sum(result)}")
    return sum(result)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python conversion.py <method> <number> <base>")
        sys.exit(1)

    method = sys.argv[1]
    n = int(sys.argv[2]) if sys.argv[2].isdigit() else sys.argv[2]
    base = int(sys.argv[3])

    if method == "divide":
        success_div(n, base)
    elif method == "multiply":
        weighted_multiply(n, base)
    else:
        print(
            "Invalid method. Use 1 for division method or 2 for weighted multiply method."
        )
