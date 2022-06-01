#date: 2022-06-01T17:06:50Z
#url: https://api.github.com/gists/015cd686f22f93de8b872e2b9591c662
#owner: https://api.github.com/users/farsil

import sys
from math import floor, ceil, inf, isinf
from dataclasses import dataclass
from argparse import ArgumentParser


@dataclass
class Boundaries:
    min: int
    max: int


class DiracFunction:
    def __init__(self, deltas=None):
        self.deltas = deltas if deltas else {}
        self.boundaries = Boundaries(min(self.deltas.keys()), max(self.deltas.keys()))

    def __call__(self, x):
        return self.deltas[x] if x in self.deltas else 0.0

    def __str__(self):
        return str(self.deltas)

    def points(self):
        return self.deltas.items()

    def squash(self, begin, end):
        begin = self.boundaries.min if isinf(begin) and begin < 0 else begin
        end = self.boundaries.max if isinf(end) and end > 0 else end

        squashed = {x: y for x, y in self.deltas.items() if begin <= x <= end}
        if begin != self.boundaries.min:
            squashed[begin] = 0
            for x in range(self.boundaries.min, 1 + begin):
                squashed[begin] += self(x)
        if end != self.boundaries.max:
            squashed[end] = 0
            for x in range(end, 1 + self.boundaries.max):
                squashed[end] += self(x)
        return DiracFunction(squashed)

    def shift(self, offset):
        return DiracFunction({x + offset: y for x, y in self.deltas.items()})

    def convolve(self, other: 'DiracFunction') -> 'DiracFunction':
        result = {}
        min_result_x = self.boundaries.min + other.boundaries.min
        max_result_x = self.boundaries.max + other.boundaries.max
        for result_x in range(min_result_x, max_result_x + 1):
            # This only works for functions made exclusively of dirac deltas
            area = 0.0
            for self_x in self.deltas.keys():
                area += self(self_x) * other(result_x - self_x)
            if area != 0.0:
                result[result_x] = area
        return DiracFunction(result)


base_pdf = DiracFunction({
    -5: 1 / 36,
    -4: 2 / 36,
    -3: 3 / 36,
    -2: 4 / 36,
    -1: 5 / 36,
    0: 6 / 36,
    1: 5 / 36,
    2: 4 / 36,
    3: 3 / 36,
    4: 2 / 36,
    5: 1 / 36
})


def removal_outcomes(amount, attempts, mod_diff):
    pdf = base_pdf.shift(mod_diff).squash(0, inf)
    for _ in range(0, attempts - 1):
        pdf = pdf.convolve(pdf)
    return pdf.squash(0, amount)


def main(amount, attempts, mod_diff):
    print(f"Remove at least {amount} influence in {attempts} attempts, with a modifier difference of {mod_diff}")
    print()
    print("Influence removed")
    points = list(removal_outcomes(amount, attempts, mod_diff).points())
    for idx, (x, y) in enumerate(points):
        amount = f"{x}:" if idx < len(points) - 1 else f"{x}+:"
        probability = f"{(100 * y):.2f}%"
        print(amount, probability)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--amount', '-a', type=int, required=True)
    parser.add_argument('--attempts', '-n', type=int, default=1)
    parser.add_argument('--mod-diff', '-m', type=int, default=0)
    main(**vars(parser.parse_args()))
