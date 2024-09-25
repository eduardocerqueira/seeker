#date: 2024-09-25T16:53:19Z
#url: https://api.github.com/gists/15def78c453113103c363c718aabacf1
#owner: https://api.github.com/users/esmitt

from pathlib import PurePath, Path


def stringify_file(filename: str) -> str:
    single_file = ""
    with open(filename) as file:
        for line in file:
            if len(line.strip()) > 0:
                line = line.split('#')
                if len(line) == 1:  # the line has comments, then ignore them
                    single_file += line[0]
    return single_file


def single_str_for_python_files(directory: PurePath) -> str:
    result: str = ""
    for filename in Path(directory).rglob("*.py"):
        if filename.name != "__init__.py":
            result += f"{filename.relative_to(filename.parent.parent)}:\n"
            result += stringify_file(str(filename.absolute()))
    return result


print(single_str_for_python_files(Path.cwd() / "app"))
