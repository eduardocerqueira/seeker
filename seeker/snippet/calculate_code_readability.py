#date: 2023-11-06T17:01:08Z
#url: https://api.github.com/gists/60de22b09e9c91a16443519a1741e0c0
#owner: https://api.github.com/users/pszemraj

import re
from itertools import chain


def calculate_readability(code_string:str) -> float:
    code = code_string.splitlines()

    # Heuristic 1: Line length
    max_line_length = 80
    long_lines = sum(1 for line in code if len(line) > max_line_length)
    long_line_ratio = long_lines / len(code)

    # Heuristic 2: Identifier length
    min_identifier_length = 2
    max_identifier_length = 20
    identifiers = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", " ".join(code))
    short_identifiers = sum(1 for id in identifiers if len(id) < min_identifier_length)
    long_identifiers = sum(1 for id in identifiers if len(id) > max_identifier_length)
    bad_identifier_ratio = (
        (short_identifiers + long_identifiers) / len(identifiers) if identifiers else 0
    )

    # Heuristic 3: Comment density
    target_comment_density = 0.15
    comment_lines = sum(1 for line in code if re.search(r"//|/\*|\*/|#", line))
    comment_density = abs(comment_lines / len(code) - target_comment_density)

    # Heuristic 4: Cyclomatic Complexity (normalized by the number of functions)
    control_structures = re.findall(
        r"\b(if|else|for|while|switch|case|default|continue|break)\b",
        " ".join(code),
        re.I,
    )
    functions = re.findall(r"\b(def|function|func|sub)\b", " ".join(code), re.I)
    cyclomatic_complexity = (len(control_structures) + 1) / (len(functions) + 1)

    # Heuristic 5: Indentation consistency
    indentation_levels = [
        len(re.match(r"^[\s\t]*", line).group()) for line in code if line.strip() != ""
    ]
    inconsistent_indentation = sum(
        1
        for i in range(1, len(indentation_levels))
        if indentation_levels[i] - indentation_levels[i - 1] not in {0, 1, -1}
    )
    indentation_inconsistency_ratio = (
        inconsistent_indentation / (len(indentation_levels) - 1)
        if len(indentation_levels) > 1
        else 0
    )
    # Normalize heuristic scores
    normalized_scores = {
        "long_line_ratio": 1 - min(long_line_ratio, 1),
        "bad_identifier_ratio": 1 - min(bad_identifier_ratio, 1),
        "comment_density_deviation": 1 - min(comment_density, 1),
        "normalized_cyclomatic_complexity": 1 / (1 + cyclomatic_complexity),
        "indentation_inconsistency_ratio": 1 - min(indentation_inconsistency_ratio, 1),
    }

    # Calculate the aggregate score as the average of the normalized scores
    aggregate_score = sum(normalized_scores.values()) / len(normalized_scores)

    return aggregate_score


# Example usage:
code_example = """def calculate_readability(code):
    # This function calculates readability
    avg_line_length = sum(len(line) for line in code) / len(code)
    return avg_line_length"""

readability_score = calculate_readability(code_example)
print(readability_score)