#date: 2024-12-26T17:07:32Z
#url: https://api.github.com/gists/9eec0a816966f131ff4299c129abbbeb
#owner: https://api.github.com/users/l1asis

"""
File: cyclic_characters.py
Author: Volodymyr Horshenin
Description: Script to convert index to a string
             of cyclic characters and vice versa.
"""


def index_to_string(index: int = 1, charset: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> str:
    """
    Converts a 1-based index to a string of cyclic characters (base: the length of the character set).

    Example:
    ("A", "B", "C", ..., "Z", "AA", "AB", "AC", ...)

    Concept:
    ```
    703 = "AAA"
        (703-1) % 26 = 0 = "A" | (703-1) // 26 = 27
         (27-1) % 26 = 0 = "A" |  (27-1) // 26 = 1
          (1-1) % 26 = 0 = "A" |   (1-1) // 26 = 0
    ```
    """
    string = ""
    while index:
        index, remainder = divmod((index - 1), len(charset))
        string = charset[remainder] + string
    return string


def string_to_index(string: str, charset: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> int:
    """
    Converts a string (base: the length of the character set) to decimal with a 1-based index.

    Concept:
    ```
    "UKR" = 14500
        "U" = 21 | "K" = 11 | "R" = 18
        21 * 26^2 + 11 * 26^1 + 18 * 26^0 = 14500
    ```
    """
    index = 0
    table = {char: i for i, char in enumerate(charset, 1)}
    for power, char in enumerate(string[::-1], 0):
        index += table[char] * len(charset) ** power
    return index
