#date: 2025-08-26T17:10:48Z
#url: https://api.github.com/gists/b4de12f7d111403758d51bf8c8866bb7
#owner: https://api.github.com/users/gxxk-dev

# AGPL v3(orlater) / By Frez79

#!/usr/bin/env python3
import sys
import os
import json
import hashlib
import toml
import argparse
from pathlib import Path


def calculate_file_hash(file_path, prefix=""):
    """Calculate SHA256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"{prefix}Error calculating hash for {file_path}: {e}")
        return None


def extract_dependencies_from_pyproject(pyproject_path, prefix=""):
    """Extract dependencies from pyproject.toml."""
    try:
        with open(pyproject_path, 'r', encoding='utf-8') as f:
            data = toml.load(f)
        
        deps = []
        project = data.get('project', {})
        
        # Regular dependencies only (skip micropython-specific ones for now)
        if 'dependencies' in project:
            for dep in project['dependencies']:
                deps.append([dep, "latest"])
        
        # Skip micropython dependencies as they may not be properly packaged
        # optional_deps = project.get('optional-dependencies', {})
        # if 'micropython' in optional_deps:
        #     for dep in optional_deps['micropython']:
        #         deps.append([dep, "latest"])
        
        return deps
    except Exception as e:
        print(f"{prefix}Error reading pyproject.toml: {e}")
        return []


def find_files_by_extensions(base_path, extensions=None):
    """Find all files with specified extensions in the project."""
    if extensions is None:
        extensions = ['.py']
    
    # Ensure extensions start with dot
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
    
    found_files = []
    base_path = Path(base_path)
    
    for ext in extensions:
        pattern = f"*{ext}"
        for file_path in base_path.rglob(pattern):
            if file_path.is_file():
                rel_path = file_path.relative_to(base_path)
                found_files.append(str(rel_path))
    
    return found_files


def generate_package_json(base_path=".", prefix="", extensions=None, pyproject_path=None):
    """Generate package.json for MicroPython package."""
    base_path = Path(base_path).resolve()
    
    if not base_path.exists():
        print(f"{prefix}Error: Path {base_path} does not exist")
        return None
    
    # Find files with specified extensions
    target_files = find_files_by_extensions(base_path, extensions)
    
    # Generate hashes
    hashes = []
    
    for file_path in target_files:
        full_path = base_path / file_path
        file_hash = calculate_file_hash(full_path, prefix)
        if file_hash:
            # Use forward slashes for paths
            file_path_normalized = str(file_path).replace('\\', '/')
            hashes.append([file_path_normalized, file_hash])
    
    # Add main files if they exist
    main_files = ["main.py", "README.md", "config.template.yaml"]
    for main_file in main_files:
        main_path = base_path / main_file
        if main_path.exists():
            if main_file.endswith('.py'):
                file_hash = calculate_file_hash(main_path, prefix)
                if file_hash:
                    hashes.append([main_file, file_hash])
    
    # Extract dependencies
    if pyproject_path:
        pyproject_path = Path(pyproject_path).resolve()
    else:
        pyproject_path = base_path / "pyproject.toml"
    
    deps = extract_dependencies_from_pyproject(pyproject_path, prefix) if pyproject_path.exists() else []
    
    # Create package.json structure
    package_data = {
        "hashes": hashes,
        "deps": deps
    }
    
    return package_data


def main():
    """Main function to handle command line arguments and generate package.json."""
    parser = argparse.ArgumentParser(
        description="Generate package.json for MicroPython package from local files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python 2packagejson.py
  python 2packagejson.py --prefix "[PKG] "
  python 2packagejson.py --extensions py yaml txt
  python 2packagejson.py /path/to/local/repo --prefix "INFO: "
  python 2packagejson.py --output ./output/package.json
  python 2packagejson.py --output ./output/ --prefix "[PKG] "
  python 2packagejson.py --pyproject /path/to/custom/pyproject.toml
  python 2packagejson.py --pyproject ../other_project/pyproject.toml --output ./custom_package.json"""
    )
    
    parser.add_argument("local_path", nargs='?', default=".", help="Local path to scan for files (default: current directory)")
    parser.add_argument("--output", "-o", help="Output path for package.json (default: package.json in current working directory). If directory, outputs package.json there; if file, outputs to that file")
    parser.add_argument("--prefix", help="Prefix to add to all output messages (default: auto-set based on path)")
    parser.add_argument("--extensions", "-e", nargs="+", default=["py"], 
                        help="File extensions to include (default: py). Can specify multiple: -e py yaml txt")
    parser.add_argument("--pyproject", "-p", help="Path to pyproject.toml file (default: pyproject.toml in the local_path)")
    
    args = parser.parse_args()
    
    # Auto-set prefix based on path if not specified
    if args.prefix is None:
        path_name = os.path.basename(os.path.abspath(args.local_path))
        args.prefix = f"[{path_name}] " if path_name != "." else ""
    
    print(f"{args.prefix}Generating package.json from local path: {os.path.abspath(args.local_path)}")
    if args.pyproject:
        print(f"{args.prefix}Using custom pyproject.toml from: {os.path.abspath(args.pyproject)}")
    
    package_data = generate_package_json(args.local_path, args.prefix, args.extensions, args.pyproject)
    
    if package_data is None:
        print(f"{args.prefix}Failed to generate package.json")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if output_path.is_dir() or (not output_path.exists() and not output_path.suffix):
            # If it's a directory (existing) or looks like a directory (no extension), append package.json
            output_path = output_path / "package.json"
    else:
        # Default: package.json in current working directory
        output_path = Path.cwd() / "package.json"
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(package_data, f, indent=2, ensure_ascii=False)
        
        print(f"{args.prefix}Successfully generated package.json at: {output_path}")
        print(f"{args.prefix}Found {len(package_data['hashes'])} files with hashes")
        print(f"{args.prefix}Found {len(package_data['deps'])} dependencies")
        
    except Exception as e:
        print(f"{args.prefix}Error writing package.json: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()