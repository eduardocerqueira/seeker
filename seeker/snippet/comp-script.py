#date: 2025-11-26T16:43:51Z
#url: https://api.github.com/gists/f2328ab948b23ebcf7507de0460f7ee1
#owner: https://api.github.com/users/elitelinuxuser

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional
from datetime import datetime

DEFAULT_ENVIRONMENTS = ["dev", "sit", "uat", "preprod"]
DEFAULT_BASE_TEMPLATE = "/CTRLFW/ADF/DF/{env}/daas/sql"


def scan_environment(env: str, base_template: str, prefix: str) -> Set[str]:
    """
    Scan a single environment directory for files named {prefix}*.json.

    Returns a set of filenames (e.g. 'myfile.json') found in that env.
    """
    dir_path = Path(base_template.format(env=env))

    if not dir_path.exists():
        print(f"[WARN] Directory does not exist for env '{env}': {dir_path}", file=sys.stderr)
        return set()

    if not dir_path.is_dir():
        print(f"[WARN] Path is not a directory for env '{env}': {dir_path}", file=sys.stderr)
        return set()

    pattern = f"{prefix}*.json"
    files = {p.name for p in dir_path.glob(pattern) if p.is_file()}

    print(f"[INFO] Env '{env}': found {len(files)} file(s) matching '{pattern}' in {dir_path}", file=sys.stderr)
    return files


def parse_base_and_dt(filename: str) -> Tuple[str, Optional[datetime]]:
    """
    Split filename into base and datetime.

    Assumes pattern: <base>_<YYYYMMDD-HHMMSSAM/PM>.json
    where <base> itself can contain underscores.

    Returns:
        base_name, datetime_or_None
    """
    stem = Path(filename).stem  # strip .json
    if "_" not in stem:
        return stem, None

    base, suffix = stem.rsplit("_", 1)

    # Clean any stray dots, just in case (e.g. "PM.")
    suffix = suffix.rstrip(".")

    # Try to parse datetime
    dt = None
    for fmt in ("%Y%m%d-%I%M%S%p", "%Y%m%d-%H%M%S"):  # 12h w/AMPM, or 24h as fallback
        try:
            dt = datetime.strptime(suffix, fmt)
            break
        except ValueError:
            continue

    if dt is None:
        print(f"[WARN] Could not parse datetime from suffix '{suffix}' in file '{filename}'", file=sys.stderr)

    return base, dt


def build_latest_diff(prefix: str,
                      environments: List[str],
                      base_template: str) -> Dict[str, Dict[str, bool]]:
    """
    Build a diff map but only for the latest timestamp per base filename.

    Steps:
    - Collect all filenames in each env.
    - Group by base name (everything before last underscore).
    - For each base, pick the filename with the latest parsed datetime.
    - For each of those latest filenames, check presence in each env.
    """
    env_to_files: Dict[str, Set[str]] = {}
    for env in environments:
        env_to_files[env] = scan_environment(env, base_template, prefix)

    # Determine latest filename for each base (across all envs)
    latest_by_base: Dict[str, Tuple[datetime, str]] = {}  # base -> (dt, filename)

    for env, files in env_to_files.items():
        for fname in files:
            base, dt = parse_base_and_dt(fname)
            if dt is None:
                continue  # skip ones we can't parse
            if base not in latest_by_base or dt > latest_by_base[base][0]:
                latest_by_base[base] = (dt, fname)

    print(f"[INFO] Unique base names with parsable timestamps: {len(latest_by_base)}", file=sys.stderr)

    # Build presence matrix for ONLY the latest filenames
    diff: Dict[str, Dict[str, bool]] = {}
    for base, (dt, latest_fname) in sorted(latest_by_base.items(), key=lambda x: x[0]):
        diff[latest_fname] = {}
        for env in environments:
            diff[latest_fname][env] = latest_fname in env_to_files.get(env, set())

    return diff


def write_csv(diff: Dict[str, Dict[str, bool]],
              environments: List[str],
              output_path: Path) -> None:
    """
    Write the diff to a CSV file with columns:
        filename, dev, sit, uat, preprod
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        header = ["filename"] + environments
        writer.writerow(header)

        for filename, env_map in diff.items():
            row = [filename]
            for env in environments:
                row.append("Found" if env_map.get(env, False) else "Not found")
            writer.writerow(row)

    print(f"[INFO] CSV written to {output_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latest timestamped JSON files across environments."
    )
    parser.add_argument(
        "prefix",
        help="File prefix to search for (matches {prefix}*.json)"
    )
    parser.add_argument(
        "-o", "--output",
        default="env_file_diff_latest.csv",
        help="Output CSV file path (default: env_file_diff_latest.csv)"
    )
    parser.add_argument(
        "--envs",
        default="dev,sit,uat,preprod",
        help="Comma-separated list of environment names "
             "(default: dev,sit,uat,preprod)"
    )
    parser.add_argument(
        "--base-template",
        default=DEFAULT_BASE_TEMPLATE,
        help="Base directory template with {env} placeholder. "
             "Default: /CTRLFW/ADF/DF/{env}/daas/sql"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    environments = [e.strip() for e in args.envs.split(",") if e.strip()]
    prefix = args.prefix
    base_template = args.base_template
    output_path = Path(args.output)

    print(f"[INFO] Using environments: {environments}", file=sys.stderr)
    print(f"[INFO] Using base template: {base_template}", file=sys.stderr)
    print(f"[INFO] Using prefix: {prefix}", file=sys.stderr)

    diff = build_latest_diff(prefix, environments, base_template)
    write_csv(diff, environments, output_path)


if __name__ == "__main__":
    main()
