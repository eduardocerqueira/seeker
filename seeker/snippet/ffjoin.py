#date: 2023-11-08T16:39:31Z
#url: https://api.github.com/gists/64af511fe6121934cb734262d7279a2b
#owner: https://api.github.com/users/sphr2k

#!/usr/bin/env python3
"""Join MP4 video files using ffmpeg."""

import logging
import os
import re
import subprocess  # noqa: S404
import tempfile
from pathlib import Path

from pyfzf.pyfzf import FzfPrompt


def _generate_common_filename(filenames):
    """Generate common filename based on input files."""
    # Ensure that we have at least one filename in the list
    if not filenames:
        return None

    # Use the first filename to get the common filename
    common_filename = filenames[0]

    # Regular expression pattern to match the part patterns
    patterns = [
        r" \([^)]*\)",  # Matches patterns like " (*)"
        r" \(Part \d+\)",  # Matches patterns like " (Part 1)"
        r" - Part \d+",  # Matches patterns like " - Part 1"
    ]

    # Replace the matched part patterns with an empty string
    for pattern in patterns:
        common_filename = re.sub(pattern, "", common_filename)

    return common_filename


def _join_video_files(input_files):
    """Join video files using ffmpeg."""
    output_file = _generate_common_filename(input_files)

    # Create a temporary file with absolute paths
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as filelist:
        absolute_paths = [os.path.abspath(file) for file in input_files]
        formatted_lines = [f"file '{path}'\n" for path in absolute_paths]
        filelist.writelines(formatted_lines)

        try:
            input_args = [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                filelist.name,
                "-c",
                "copy",
                os.path.join(os.getcwd(), output_file),  # Set the target directory
            ]
            subprocess.run(input_args, check=True)  # noqa: S603
            logging.info(f"Files merged and saved as {output_file}")
        except subprocess.CalledProcessError as ex:
            logging.error(f"Error joining video files: {ex}")
        finally:
            # Clean up the temporary file
            os.remove(filelist.name)


def _get_video_files():
    pattern = re.compile(r"\.(mp4|m4v)$")
    files = []
    for path in Path(".").rglob("*"):
        if pattern.search(path.name):
            files.append(str(path))
    return files


def _select_video_files(files):
    """Multi-Select video files using fzf."""
    fzf = FzfPrompt()
    selected = fzf.prompt(files, "--multi --no-sort --tac")

    if len(selected) >= 2:
        selected.sort(key=lambda file: file.lower())
        logging.info(f"Joining selected video files: {selected}")
        return selected
    else:
        logging.warning("Please select at least 2 files!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",  # noqa: WPS323
    )
    video_files = _get_video_files()
    selected_files = _select_video_files(video_files)
    if selected_files:
        _join_video_files(selected_files)
