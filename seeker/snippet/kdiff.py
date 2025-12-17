#date: 2025-12-17T17:07:15Z
#url: https://api.github.com/gists/dbc3ab3382aeb3c3279a72b29d0f9e37
#owner: https://api.github.com/users/rochacon

#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def get_resource_key(resource: Dict[str, Any]) -> str:
    """Generate a unique key for a resource based on apiVersion, kind, namespace, and name."""
    api_version = resource.get("apiVersion", "")
    kind = resource.get("kind", "")
    metadata = resource.get("metadata", {})
    name = metadata.get("name", "")
    namespace = metadata.get("namespace", "")
    
    return f"{api_version}/{kind}/{namespace}/{name}"


def parse_yaml_file(file_path: str) -> Dict[str, Tuple[Dict[str, Any], str]]:
    """Parse YAML file into a dictionary of resources keyed by their unique identifier."""
    resources = {}
    
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Parse all YAML documents in the file
    documents = list(yaml.safe_load_all(content))
    
    # Split content by document separator to preserve original formatting
    yaml_parts = content.split('\n---\n')
    
    # Handle edge case where file starts with ---
    if content.startswith('---\n'):
        yaml_parts = yaml_parts[1:]
    
    for i, doc in enumerate(documents):
        if doc is None or not isinstance(doc, dict):
            continue
        
        key = get_resource_key(doc)
        
        # Get the original YAML text for this document
        if i < len(yaml_parts):
            yaml_text = yaml_parts[i].strip()
            # Re-add the separator if it's not the last document
            if not yaml_text.startswith('---'):
                yaml_text = f"---\n{yaml_text}"
        else:
            # Fallback to dumping the parsed document
            yaml_text = yaml.dump(doc, default_flow_style=False, sort_keys=False)
        
        resources[key] = (doc, yaml_text)
    
    return resources


def open_vimdiff(yaml1: str, yaml2: str, key: str, temp_dir: Path) -> None:
    """Open nvim in diff mode for two YAML strings."""
    # Create safe filename from key
    safe_key = key.replace("/", "_").replace(" ", "_")
    for char in '<>:"|?*':
        safe_key = safe_key.replace(char, "_")
    
    file1 = temp_dir / f"{safe_key}_file1.yaml"
    file2 = temp_dir / f"{safe_key}_file2.yaml"
    
    # Write temporary files
    file1.write_text(yaml1)
    file2.write_text(yaml2)
    
    # Open nvim in diff mode
    try:
        subprocess.run(["nvim", "-d", str(file1), str(file2)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running nvim: {e}", file=sys.stderr)
    except FileNotFoundError:
        print("Error: nvim not found. Please install neovim.", file=sys.stderr)
        sys.exit(1)


def prompt_user(message: str) -> bool:
    """Prompt user for yes/no input."""
    try:
        response = input(message).strip().lower()
        return response in ('y', 'yes')
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare YAML resources between two files and open vimdiff for each difference."
    )
    parser.add_argument("file1", help="First YAML file")
    parser.add_argument("file2", help="Second YAML file")
    parser.add_argument(
        "--auto-open",
        action="store_true",
        help="Automatically open all diffs without prompting"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.file1).exists():
        print(f"Error: File not found: {args.file1}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.file2).exists():
        print(f"Error: File not found: {args.file2}", file=sys.stderr)
        sys.exit(1)
    
    # Parse YAML files
    print(f"Reading {args.file1}...")
    resources1 = parse_yaml_file(args.file1)
    
    print(f"Reading {args.file2}...")
    resources2 = parse_yaml_file(args.file2)
    
    print(f"\nFound {len(resources1)} resources in file 1")
    print(f"Found {len(resources2)} resources in file 2\n")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory(prefix="yaml-diff-") as temp_dir:
        temp_path = Path(temp_dir)
        
        # Find all unique keys
        all_keys = set(resources1.keys()) | set(resources2.keys())
        
        diff_count = 0
        only_in_file1 = 0
        only_in_file2 = 0
        identical = 0
        
        for key in sorted(all_keys):
            res1 = resources1.get(key)
            res2 = resources2.get(key)
            
            if res1 is None:
                print(f"⊕ Only in file 2: {key}")
                only_in_file2 += 1
                
                if args.auto_open or prompt_user("Open in vimdiff? [y/N]: "):
                    open_vimdiff("# Resource not in file 1\n", res2[1], key, temp_path)
                    
            elif res2 is None:
                print(f"⊖ Only in file 1: {key}")
                only_in_file1 += 1
                
                if args.auto_open or prompt_user("Open in vimdiff? [y/N]: "):
                    open_vimdiff(res1[1], "# Resource not in file 2\n", key, temp_path)
                    
            elif res1[1] != res2[1]:
                print(f"≠ Different: {key}")
                diff_count += 1
                
                if args.auto_open or prompt_user("Open in vimdiff? [y/N]: "):
                    open_vimdiff(res1[1], res2[1], key, temp_path)
            else:
                identical += 1
        
        print(f"\n=== Summary ===")
        print(f"Identical: {identical}")
        print(f"Different: {diff_count}")
        print(f"Only in file 1: {only_in_file1}")
        print(f"Only in file 2: {only_in_file2}")


if __name__ == "__main__":
    main()
