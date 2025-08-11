#date: 2025-08-11T16:56:30Z
#url: https://api.github.com/gists/8e9ff7caaf879c70258bf65f3584b1e5
#owner: https://api.github.com/users/phntxx

import random
import math

COMPUTER_NUMPAD = {
    0: [ 1, 2, 3 ],
    1: [ 0, 2, 4, 5 ],
    2: [ 0, 1, 3, 4, 5, 6 ],
    3: [ 2, 5, 6 ],
    4: [ 1, 2, 5, 7, 8 ],
    5: [ 1, 2, 3, 4, 6, 7, 8, 9 ],
    6: [ 2, 3, 5, 8, 9 ],
    7: [ 4, 5, 8 ],
    8: [ 4, 5, 6, 7, 9 ],
    9: [ 5, 6, 8 ],
}

PHONE_NUMPAD = {
    0: [ 7, 8, 9 ],
    1: [ 2, 4, 5 ],
    2: [ 1, 3, 4, 5, 6 ],
    3: [ 2, 5, 6 ],
    4: [ 1, 2, 5, 7, 8 ],
    5: [ 1, 2, 3, 4, 6, 7, 8, 9 ],
    6: [ 2, 3, 5, 8, 9 ],
    7: [ 0, 4, 5, 8 ],
    8: [ 0, 4, 5, 6, 7, 9 ],
    9: [ 0, 5, 6, 8 ],
}


def concat(elements: list[int]) -> int:
    """
    Concatenates a list of integers.

    :param elements: List of integers to concatenate.
    :type elements: list[str]

    :returns: The numbers in `digits` as a concatenated integer.
    :rtype: int
    """

    result = [
        int(item * math.pow(10, len(elements) - (1 + index)))
        for index, item in enumerate(elements)
    ]

    return sum(result)


def random_number(length: int) -> str:
    """
    Generate a random number with n digits.

    :param length: Desired digit-length of the random number 
    :type length: int

    :returns: A random number with n digits.
    :rtype: str
    """

    digits = [ random.randint(0, 9) for i in range(0, length) ]
    return concat(digits)


def walk_recursive(
    map: dict[int, list[int]],
    remaining_moves: int,
    current_position: int | None = None,
    past_moves: list[int] = []
) -> list[int]:
    """
    Recursive function to generate random numbers with a given set of digits by generating
    random routes across an unweighted, bidirectional graph.

    :param map: The map to use (`COMPUTER_NUMPAD` and `PHONE_NUMPAD` are provided above).
    :type map: dict[str, list[str]]

    :param remaining_moves: The desired count of digits in the random number.
    :type length: int

    :param current_position: The current position of the recursive algorithm. Set to a random digit to choose a starting digit.
    :type current_position: int | None

    :param past_moves: List of past moves made by the algorithm.
    :type past_moves: list[int]

    :returns: A list of random numbers with `length` elements.
    :rtype: list[int]
    """

    if current_position is None:
        current_position = random.choice(list(map.keys()))

    if remaining_moves == 0:
        return past_moves

    new_position = random.choice(map[current_position])

    while past_moves.count(new_position) >= 2:
        new_position = random.choice(map[current_position])

    return walk_recursive(
        map,
        remaining_moves - 1,
        new_position,
        [ *past_moves, new_position ]
    )


def random_number(map: dict[int, list[int]], length: int) -> str:
    """
    Simple wrapper function for `walk_recursive`.

    :param map: The map to use (`COMPUTER_NUMPAD` and `PHONE_NUMPAD` are provided above).
    :type map: dict[str, list[str]]

    :param length: The desired count of digits in the random number.
    :type length: int

    :returns: A random number with `length` digits.
    :rtype: str
    """
    new_number = walk_recursive(map, length)

    return concat(new_number)


def choose_random(length: int, iterations: int) -> str:
    """
    Generate a number of random numbers with a given
    length, then randomly select one of them. Aims to
    provide better entropy.

    :param length: The desired length of the random number
    :type length: int

    :param iterations: The number of numbers to generate and pick from
    :type iterations: int

    :returns: A random number with `length` digits.
    :rtype: int
    """

    options = [ random_number(length) for i in range(0, iterations) ]
    return random.choice(options)


if __name__ == "__main__":

    # Feel free to substitute for another method here
    a = random_number(COMPUTER_NUMPAD, 6)

    if a < 100000:
        print(f"0{a}")
    else:
        print(a)
