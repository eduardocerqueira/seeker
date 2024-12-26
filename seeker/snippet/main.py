#date: 2024-12-26T17:07:29Z
#url: https://api.github.com/gists/e5187a542c85371e97717917e41e4d72
#owner: https://api.github.com/users/mypy-play

import tempfile
import re

# you can use any of the following terms for your annotations
from pathlib import Path
from typing import Any, List, Union, Tuple, Dict, Optional
# including basic types: int, float, str, bool


def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)


def format_name(first_name: str, last_name: str) -> str:
    return f"{last_name.upper()}, {first_name.capitalize()}"


def process_data(data: Optional[List[int]]) -> List[int]:
    if not data:
        return []
    processed = [x * 2 for x in data]
    return processed


def get_user_id_from_folder_name(folder_path: Path) -> int:
    assert folder_path.is_dir(), "folder_path must be a directory"
    pattern = re.compile(r"(\d+)_")
    match = pattern.match(folder_path.name)
    if match:
        return int(match.group(1))
    return -1


def main():
    names = [("jane", "doe"), ("john", "smith")]
    formatted_names = [format_name(first, last) for first, last in names]
    print(formatted_names)

    user_id = 0
    tempdir = tempfile.mkdtemp()
    for first, last in names:
        user_id += 1
        name_folder = Path(tempdir) / f"{user_id}_{last}_{first}"
        name_folder.mkdir()
        print(f"Created folder: {name_folder}")
        user_id_inferred = get_user_id_from_folder_name(str(name_folder))
        print(f"Extracted user_id: {user_id_inferred}")

    numbers = [2.3, 1-5]
    average = calculate_average(numbers)
    print(average)

    data = [1, "2", 3]
    processed_data = process_data(data)
    print(processed_data)

    string_data = "12345"
    string_average = calculate_average(string_data)
    print(string_average)


if __name__ == "__main__":
    main()
