#date: 2022-04-13T17:15:10Z
#url: https://api.github.com/gists/7cf001dbf50b87d3479dce31a7fa0215
#owner: https://api.github.com/users/epage

#!/usr/bin/env python3

import argparse
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", metavar="COUNT", type=int, default=1000)
    parser.add_argument("--nest", metavar="COUNT", type=int, default=1)
    parser.add_argument("--inherit", action="store_true")
    parser.add_argument("--target", type=pathlib.Path, default=".")
    args = parser.parse_args()

    workspace_root = args.target

    package_root = workspace_root / "crates"
    members = []
    for index in range(args.members):
        name = f"package_{index}"
        current_root = package_root / name
        current_root.mkdir(parents=True, exist_ok=True)
        current_manifest_path = current_root / "Cargo.toml"
        current_manifest_path.write_text(package_manifest(name, args.inherit))
        current_source = current_root / "src/lib.rs"
        current_source.parent.mkdir(parents=True, exist_ok=True)
        current_source.write_text("")

        members.append(str(current_root.relative_to(workspace_root)))

        if index % args.nest == 0:
            package_root = workspace_root / "crates"
        else:
            package_root = current_root

    workspace_path = workspace_root / "Cargo.toml"
    workspace_path.write_text(workspace_manifest(members, args.inherit))


def workspace_manifest(members, inherit):
    members = '",\n    "'.join(members)
    if inherit:
        return f"""
[workspace]
version = "1.0.0"
members = [
    "{members}"
]
"""
    else:
        return f"""
[workspace]
members = [
    "{members}"
]
"""

def package_manifest(name, inherit):
    if inherit:
        return f"""
[package]
name = "{name}"
version.workspace = true
"""
    else:
        return f"""
[package]
name = "{name}"
version = "1.0.0"
"""

if __name__ == "__main__":
    main()
