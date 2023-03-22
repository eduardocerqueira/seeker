#date: 2023-03-22T16:45:07Z
#url: https://api.github.com/gists/88ddf34c07af2d8b9aa2be236ce554ef
#owner: https://api.github.com/users/cn-ml

from argparse import REMAINDER, ArgumentParser, BooleanOptionalAction
from pathlib import Path
from subprocess import run
from typing import Sequence


class ProgramException(Exception):
    pass

def parse_args():
    parser = ArgumentParser(description="Execute a command for all files in a directory.")
    parser.add_argument("-r", "--recursive", type=bool, default=False, action=BooleanOptionalAction, help="Whether to recursively select files.")
    parser.add_argument("-d", "--dry", type=bool, default=False, action=BooleanOptionalAction, help="Whether to dry run the commands.")
    parser.add_argument("-p", "--placeholder", type=str, default="FILE", help="Placeholder for the filename in the command.")
    parser.add_argument("-m", "--match", type=str, default="*", help="Pattern to match the base filename against.")
    parser.add_argument("directory", type=Path, help="Directory to select files from.")
    parser.add_argument("command", nargs=REMAINDER, metavar="CMD", help="Command to execute.")
    return parser.parse_args()

def run_shell_command(file_cmd: Sequence[str], dry: bool=False):
    if dry:
        print("Command:", file_cmd)
        return
    process = run(file_cmd, shell=True)
    if process.returncode != 0:
        raise ProgramException(f"Shell command '{' '.join(file_cmd)}' failed!")

def program(directory: Path, command: Sequence[str], placeholder: str, recursive: bool=False, dry: bool=False, pattern: str="*"):
    if not placeholder in command:
        raise ProgramException(f"Placeholder '{placeholder}' must be used in the command!")
    globber = directory.rglob if recursive else directory.glob
    entries = globber(pattern)
    files = (file for file in entries if file.is_file())
    for file in files:
        file_cmd = [str(file) if arg == placeholder else arg for arg in command]
        run_shell_command(file_cmd, dry)

def main():
    args = parse_args()
    try:
        program(args.directory, args.command, args. placeholder, args.recursive, args.dry, args.match)
    except ProgramException as e:
        print(f"Program failed: {e}")
        exit(1)
    except Exception as e:
        print(f"Unhandled exception: {e}")
        exit(2)


if __name__ == "__main__":
    main()
