#date: 2024-11-18T17:03:55Z
#url: https://api.github.com/gists/04d4b5ed6b4fcd801d7311c3400b2c0c
#owner: https://api.github.com/users/kielmarj

#!/usr/bin/env python3

"""
Code Context Consolidator

Consolidate code files from a specified directory into a single text file for LLM context.

This script traverses a given directory, collects code files, and compiles them into a single
text file with appropriate code blocks and directory structure for providing context to an LLM.
It supports inclusion and exclusion of specific file types, files, and folders.

**Running with `uv`**

To ensure that all dependencies are managed without manually handling environments, it's recommended
to run this script using [`uv`](https://github.com/astral-sh/uv). `uv` automatically manages virtual
environments and dependencies for your scripts.

Install `uv` by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

Run the script using:

    uv run --with typer code_context_consolidator.py [OPTIONS] DIRECTORY

For more details on running scripts with `uv`, see the [documentation](https://docs.astral.sh/uv/guides/scripts/).

**Usage:**

    python code_context_consolidator.py [OPTIONS] DIRECTORY

**Options:**

    --exclude-file-types TEXT     Comma-separated list of file extensions to exclude.
    --exclude-files TEXT          Comma-separated list of file names to exclude.
    --exclude-folders TEXT        Comma-separated list of folder names to exclude.
    --include-file-types TEXT     Comma-separated list of file extensions to include.
    --include-folders TEXT        Comma-separated list of folder names to include.
    --output-file TEXT            Path and name of the output text file.

**Example:**

To exclude multiple files and folders, such as `node_modules`, `dist`, `cdk.out`, `__pycache__`,
a specific folder under `src`, all Jupyter Notebook files (`*.ipynb`), and `__init__.py` files, run:

    uv run --with typer code_context_consolidator.py \
        --exclude-file-types ipynb \
        --exclude-files __init__.py \
        --exclude-folders node_modules,dist,cdk.out,__pycache__,src/specific_folder \
        path/to/your/project

This will consolidate all code files in `path/to/your/project`, excluding the specified files and folders,
into a single `output.txt` file in the current working directory.
"""

import traceback
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(add_completion=False)


def generate_directory_structure(directory: Path, exclude_folders: List[str]) -> str:
    """
    Generate the directory structure of the given directory.

    Args:
        directory (Path): The root directory to generate the structure from.
        exclude_folders (List[str]): List of folder names to exclude.

    Returns:
        str: A string representing the directory structure in tree format.
    """
    from io import StringIO

    output = StringIO()

    def tree(dir_path: Path, prefix: str = ""):
        entries = [
            entry for entry in dir_path.iterdir() if entry.name not in exclude_folders
        ]
        entries.sort()
        entries_count = len(entries)
        for index, entry in enumerate(entries):
            connector = "├── " if index < entries_count - 1 else "└── "
            output.write(f"{prefix}{connector}{entry.name}\n")
            if entry.is_dir():
                extension = "│   " if index < entries_count - 1 else "    "
                tree(entry, prefix + extension)

    output.write(f"{directory.name}\n")
    tree(directory)
    return output.getvalue()


def collect_files(
    directory: Path,
    exclude_file_types: List[str],
    exclude_folders: List[str],
    include_file_types: Optional[List[str]],
    include_folders: Optional[List[str]],
    exclude_files: List[str],
) -> List[Path]:
    """
    Collect files from the directory respecting inclusion and exclusion criteria.

    Args:
        directory (Path): The root directory to collect files from.
        exclude_file_types (List[str]): List of file extensions to exclude.
        exclude_folders (List[str]): List of folder names to exclude.
        include_file_types (Optional[List[str]]): List of file extensions to include.
        include_folders (Optional[List[str]]): List of folder names to include.
        exclude_files (List[str]): List of file names to exclude.

    Returns:
        List[Path]: A list of file paths collected.
    """
    files = []
    for path in directory.rglob("*"):
        if path.is_file():
            if any(folder in path.parts for folder in exclude_folders):
                continue
            if include_folders and not any(
                folder in path.parts for folder in include_folders
            ):
                continue
            if include_file_types and path.suffix not in include_file_types:
                continue
            if path.suffix in exclude_file_types:
                continue
            if path.name in exclude_files:
                continue
            files.append(path)
    return files


def get_code_block_language(file_extension: str) -> str:
    """
    Map file extensions to code block languages.

    Args:
        file_extension (str): The file extension.

    Returns:
        str: The code block language identifier.
    """
    extension_language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".html": "html",
        ".css": "css",
        ".md": "markdown",
        ".sh": "shell",
        ".json": "json",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".xml": "xml",
        ".go": "go",
        ".rb": "ruby",
        ".php": "php",
        ".rs": "rust",
    }
    return extension_language_map.get(file_extension, "")


@app.command()
def main(
    directory: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory to parse and compile the code from.",
    ),
    exclude_file_types: Optional[str] = typer.Option(
        None, help="Comma-separated list of file extensions to exclude."
    ),
    exclude_files: Optional[str] = typer.Option(
        None, help="Comma-separated list of file names to exclude."
    ),
    exclude_folders: Optional[str] = typer.Option(
        None, help="Comma-separated list of folder names to exclude."
    ),
    include_file_types: Optional[str] = typer.Option(
        None, help="Comma-separated list of file extensions to include."
    ),
    include_folders: Optional[str] = typer.Option(
        None, help="Comma-separated list of folder names to include."
    ),
    output_file: Optional[Path] = typer.Option(
        None, help="Path and name of the output text file."
    ),
):
    """
    Consolidate code files from a specified directory into a single text file for LLM context.
    """
    try:
        if not output_file:
            output_file = Path.cwd() / "output.txt"

        exclude_file_types_list = (
            exclude_file_types.split(",") if exclude_file_types else []
        )
        exclude_file_types_list = [
            f".{ext.strip().lstrip('.')}" for ext in exclude_file_types_list
        ]

        exclude_files_list = exclude_files.split(",") if exclude_files else []
        exclude_files_list = [name.strip() for name in exclude_files_list]

        exclude_folders_list = exclude_folders.split(",") if exclude_folders else []

        include_file_types_list = (
            include_file_types.split(",") if include_file_types else None
        )
        if include_file_types_list:
            include_file_types_list = [
                f".{ext.strip().lstrip('.')}" for ext in include_file_types_list
            ]

        include_folders_list = include_folders.split(",") if include_folders else None

        directory_structure = generate_directory_structure(
            directory, exclude_folders_list
        )
        content_lines = ["```shell", directory_structure, "```", "\n"]

        files = collect_files(
            directory,
            exclude_file_types_list,
            exclude_folders_list,
            include_file_types_list,
            include_folders_list,
            exclude_files_list,
        )
        for file_path in sorted(files):
            relative_path = file_path.relative_to(directory)
            language = get_code_block_language(file_path.suffix)
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                file_content = f.read()
            content_lines.append(f"File: {relative_path}")
            content_lines.append(f"```{language}")
            content_lines.append(file_content)
            content_lines.append("```")
            content_lines.append("\n")

        output_file.write_text("\n".join(content_lines), encoding="utf-8")

        typer.echo(f"Consolidated code written to {output_file}")

    except Exception as e:
        typer.echo(f"An error occurred: {e}")
        typer.echo(traceback.format_exc())


if __name__ == "__main__":
    app()
