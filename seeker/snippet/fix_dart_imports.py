#date: 2025-10-16T17:08:23Z
#url: https://api.github.com/gists/fa5f8d32bec4612900131a88bed93395
#owner: https://api.github.com/users/Iamstanlee

#!/usr/bin/env python3
import os
import re
from pathlib import Path

def get_package_import_path(file_path, lib_root):
    """Convert a file path to a package import path."""
    rel_path = os.path.relpath(file_path, lib_root)
    return f"package:aaj_core/{rel_path.replace(os.sep, '/')}"

def resolve_relative_import(current_file, relative_import, lib_root):
    """Resolve a relative import to a package import."""
    current_dir = os.path.dirname(current_file)

    # Remove leading ./ if present
    if relative_import.startswith('./'):
        relative_import = relative_import[2:]

    # Handle ../ paths
    parts = relative_import.split('/')
    target_dir = current_dir

    i = 0
    while i < len(parts) and parts[i] == '..':
        target_dir = os.path.dirname(target_dir)
        i += 1

    # Remaining parts are the path to the file
    remaining_path = '/'.join(parts[i:])
    target_file = os.path.join(target_dir, remaining_path)

    # If it doesn't have .dart extension, add it
    if not target_file.endswith('.dart'):
        target_file += '.dart'

    # Convert to package import
    rel_to_lib = os.path.relpath(target_file, lib_root)
    package_import = f"package:aaj_core/{rel_to_lib.replace(os.sep, '/')}"

    return package_import

def fix_imports_in_file(file_path, lib_root):
    """Fix all relative imports in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    lines = content.split('\n')
    modified_lines = []
    changes = []

    for line in lines:
        # Match import statements with relative paths
        match = re.match(r'^(import\s+[\'"])(\.\./|\./)([^\'"]+)([\'"];?)(.*)$', line)

        if match:
            prefix = match.group(1)  # import '
            rel_indicator = match.group(2)  # ../ or ./
            rel_path = match.group(3)  # rest of the path
            quote_end = match.group(4)  # ';
            comment = match.group(5)  # any trailing comment

            full_rel_path = rel_indicator + rel_path
            package_import = resolve_relative_import(file_path, full_rel_path, lib_root)

            new_line = f"{prefix}{package_import}{quote_end}{comment}"
            modified_lines.append(new_line)
            changes.append(f"  {line.strip()} -> {new_line.strip()}")
        else:
            modified_lines.append(line)

    if changes:
        new_content = '\n'.join(modified_lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return changes

    return None

def main():
    script_dir = Path(__file__).parent
    lib_root = script_dir / 'lib'
    scan_root = lib_root / 'features' / 'scan'

    print(f"Scanning directory: {scan_root}")
    print(f"Lib root: {lib_root}")
    print()

    # Find all .dart files (excluding generated files)
    dart_files = []
    for root, dirs, files in os.walk(scan_root):
        for file in files:
            if file.endswith('.dart') and not file.endswith('.g.dart') and not file.endswith('.freezed.dart'):
                dart_files.append(os.path.join(root, file))

    print(f"Found {len(dart_files)} Dart files to process")
    print()

    total_files_changed = 0
    total_imports_fixed = 0

    for dart_file in sorted(dart_files):
        changes = fix_imports_in_file(dart_file, lib_root)
        if changes:
            total_files_changed += 1
            total_imports_fixed += len(changes)
            rel_file = os.path.relpath(dart_file, script_dir)
            print(f"Fixed {len(changes)} import(s) in: {rel_file}")
            for change in changes:
                print(change)
            print()

    print(f"\nSummary:")
    print(f"  Files processed: {len(dart_files)}")
    print(f"  Files changed: {total_files_changed}")
    print(f"  Imports fixed: {total_imports_fixed}")

if __name__ == '__main__':
    main()