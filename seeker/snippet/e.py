#date: 2025-07-24T16:48:38Z
#url: https://api.github.com/gists/f51fa200fb2d8abe6daf9ac241e17af9
#owner: https://api.github.com/users/TheCrazyGM

#!/usr/bin/env python3
import argparse
import os
import re
import sys


def main():
    # Determine editor
    editor_env = os.getenv("EDITOR")
    editor = editor_env.split() if editor_env else ["/usr/bin/sensible-editor"]

    parser = argparse.ArgumentParser(
        description="Open editor at specific file locations and/or parse stdin for filenames.",
        epilog="Examples:\n  e.py foo.py:10 bar.txt\n  some_command | e.py -s",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files (optionally with :line) to open, e.g. foo.py:42",
    )
    parser.add_argument(
        "-s",
        "--stdin",
        action="store_true",
        help="Parse standard input for filename[:line] patterns.",
    )
    args = parser.parse_args()

    # Dictionary to store {filename: line_number}
    file_lines = {}

    # Process command line arguments
    for arg in args.files:
        filename, line = parse_file_arg(arg)
        file_lines[filename] = line

    # Process stdin if requested or if stdin is not a TTY
    parse_stdin = args.stdin or (not sys.stdin.isatty())
    if parse_stdin:
        for line in sys.stdin:
            # Try Python traceback format
            m = re.search(r'File "([^"]+)", line (\d+)', line)
            if m:
                filename = find_existing_path(m.group(1))
                file_lines[filename] = int(m.group(2))
                continue

            # Try colon format (file.py:123)
            m = re.search(r'([^:\s\t]+):(\d+)', line)
            if m:
                filename = find_existing_path(m.group(1))
                file_lines[filename] = int(m.group(2))

        # Reopen stdin as /dev/tty for the editor
        try:
            sys.stdin = open("/dev/tty")
        except Exception as e:
            print(f"Can't open /dev/tty: {e}", file=sys.stderr)
            sys.exit(1)

    # No files to edit?
    if not file_lines:
        print("No files to edit!", file=sys.stderr)
        sys.exit(1)

    # Build editor command
    cmd = build_editor_command(editor, file_lines)
    
    # Execute editor
    os.execvp(cmd[0], cmd)


def find_existing_path(path):
    """Find an existing version of the path, handling symlinks."""
    # Try different path variations
    paths_to_try = [
        path,                     # Original path
        os.path.abspath(path),    # Absolute path
        os.path.realpath(path)    # Path with symlinks resolved
    ]
    
    # Use the first path that exists
    for p in paths_to_try:
        if os.path.exists(p):
            return p
            
    # If no path exists, return the absolute path
    return os.path.abspath(path)


def parse_file_arg(arg):
    """Parse a file argument like 'file.py:123' into (filename, line_number)."""
    # Remove trailing colon (common in stack traces)
    if arg.endswith(':'):
        arg = arg[:-1]

    # Check for line number
    m = re.search(r':(\d+)(?::\d+)?$', arg)
    if m:
        line_num = int(m.group(1))
        # Remove the line number part
        filename = re.sub(r':\d+(?::\d+)?$', '', arg)
        return find_existing_path(filename), line_num
    else:
        return find_existing_path(arg), None


def build_editor_command(editor, file_lines):
    """Build the editor command with proper line number positioning."""
    cmd = list(editor)
    
    # Collect all existing files
    valid_files = [(f, l) for f, l in file_lines.items() if os.path.exists(f)]
    
    if not valid_files:
        print("No valid files found!", file=sys.stderr)
        sys.exit(1)
    
    # Add the first file
    first_file, first_line = valid_files[0]
    if first_line:
        cmd.extend(["+call cursor(" + str(first_line) + ",0)", first_file])
    else:
        cmd.append(first_file)
    
    # Add additional files
    for filename, line in valid_files[1:]:
        if line:
            cmd.extend(["-c", f"badd {filename}", "-c", f"buffer {os.path.basename(filename)}", 
                       "-c", f"call cursor({line},0)"])
        else:
            cmd.extend(["-c", f"badd {filename}"])
    
    return cmd


if __name__ == "__main__":
    main()
