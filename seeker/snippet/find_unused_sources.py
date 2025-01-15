#date: 2025-01-15T17:05:42Z
#url: https://api.github.com/gists/20ca11e57cef0bc996bf73f6bf96f7b0
#owner: https://api.github.com/users/charlielewisme

import json
import subprocess
from pathlib import Path


def get_unused_sources():
    """Find unused sources using dbt's manifest"""

    # Always run dbt parse first
    print("Running dbt parse to generate fresh manifest...")
    subprocess.run("dbt parse", shell=True, check=True)

    # Load the manifest
    with open("target/manifest.json") as f:
        manifest = json.load(f)

    # Get all sources
    sources = {node_id: node for node_id, node in manifest["sources"].items()}

    # Check which sources have no child nodes
    unused_sources = {
        node_id: node
        for node_id, node in sources.items()
        if not manifest["child_map"].get(node_id, [])
    }

    return unused_sources, len(sources)


def main():
    unused_sources, total_sources = get_unused_sources()

    # Format and write results to file
    with open("unused_sources.txt", "w") as f:
        for node in sorted(
            unused_sources.values(), key=lambda x: f"{x['source_name']}.{x['name']}"
        ):
            source_name = node["source_name"]
            table_name = node["name"]
            f.write(f"{source_name}.{table_name}\n")

    # Print summary to console
    print(f"\nSummary:")
    print(f"Total sources: {total_sources}")
    print(f"Unused sources: {len(unused_sources)}")
    print(f"\nResults have been written to unused_sources.txt")


if __name__ == "__main__":
    main()
