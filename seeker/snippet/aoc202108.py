#date: 2021-12-08T16:58:54Z
#url: https://api.github.com/gists/447f3898fb83fa3fff9221aa29a285b0
#owner: https://api.github.com/users/grovduck

"""AoC 8, 2021"""

# Standard library imports
import pathlib
import sys
from itertools import permutations


def parse(puzzle_input):
    """Parse input"""
    inputs = []
    outputs = []
    for line in puzzle_input.split("\n"):
        input_str, output_str = line.split(" | ")
        inputs.append([x for x in input_str.split()])
        outputs.append([x for x in output_str.split()])
    return inputs, outputs


def part1(data):
    """Solve part 1"""
    count = 0
    for output in data[1]:
        for el in output:
            if len(el) != 5 and len(el) != 6:
                count += 1
    return count


def part2(data):
    """Solve part 2"""

    # Binary encoding of segments to "light up" for each digit
    NUMS = [
        [1, 1, 1, 0, 1, 1, 1],  # 0
        [0, 0, 1, 0, 0, 1, 0],  # 1
        [1, 0, 1, 1, 1, 0, 1],  # 2
        [1, 0, 1, 1, 0, 1, 1],  # 3
        [0, 1, 1, 1, 0, 1, 0],  # 4
        [1, 1, 0, 1, 0, 1, 1],  # 5
        [1, 1, 0, 1, 1, 1, 1],  # 6
        [1, 0, 1, 0, 0, 1, 0],  # 7
        [1, 1, 1, 1, 1, 1, 1],  # 8
        [1, 1, 1, 1, 0, 1, 1],  # 9
    ]
    total = 0
    for inputs, outputs in zip(data[0], data[1]):
        for p in permutations("abcdefg"):
            found_solution = True

            # Create the binary list of each word based on this permutation
            # and verify that each word produces a valid number.
            for word in inputs:
                code = [0] * 7
                for letter in word:
                    code[p.index(letter)] = 1
                if code not in NUMS:
                    found_solution = False
                    break

            # All numbers were valid, use this permutation as the mapping
            if found_solution:
                solution = p

        # Apply the mapping to each word in the output list
        output = ""
        for word in outputs:
            code = [0] * 7
            for letter in word:
                code[solution.index(letter)] = 1
            output += str(NUMS.index(code))
        total += int(output)
    return total


def solve(puzzle_input):
    """Solve the puzzle for the given input"""
    data = parse(puzzle_input)
    solution1 = part1(data)
    solution2 = part2(data)

    return solution1, solution2


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(f"\n{path}:")
        solutions = solve(puzzle_input=pathlib.Path(path).read_text().strip())
        print("\n".join(str(solution) for solution in solutions))
