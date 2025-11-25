#date: 2025-11-25T16:56:30Z
#url: https://api.github.com/gists/3d16b26ef7f14494bc8231e99ebe4a1b
#owner: https://api.github.com/users/itlackey

#!/usr/bin/env python3
"""Copy a manifest subtree and all referenced blobs into another Ollama store."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a directory under manifests/ plus every blob referenced by"
            " the manifest JSON into another Ollama installation."
        )
    )
    parser.add_argument(
        "manifest_path",
        help=(
            "Path to either a manifest directory or a specific manifest file, "
            "including the leading manifests/ segment"
        ),
    )
    parser.add_argument(
        "target_root",
        help="Destination root that should/will contain manifests/ and blobs/",
    )
    return parser.parse_args()


def resolve_source_manifest(manifest_path: str, source_manifests: Path) -> Path:
    source_path = Path(manifest_path).expanduser().resolve()
    try:
        source_path.relative_to(source_manifests)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(
            f"Manifest path must stay within {source_manifests}, got {source_path}"
        ) from exc
    if not source_path.exists():
        raise FileNotFoundError(f"Manifest path does not exist: {source_path}")
    return source_path


def copy_manifest_payload(source_path: Path, target_path: Path) -> None:
    if source_path.is_dir():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def extract_digests(payload: object) -> Iterable[str]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "digest" and isinstance(value, str):
                yield value
            else:
                yield from extract_digests(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from extract_digests(item)


def gather_digests(manifest_path: Path) -> Set[str]:
    digests: Set[str] = set()
    files: Iterable[Path]
    if manifest_path.is_dir():
        files = (p for p in manifest_path.rglob("*") if p.is_file())
    else:
        files = (manifest_path,)
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")
            payload = json.loads(content)
        except (OSError, json.JSONDecodeError):
            continue
        digests.update(extract_digests(payload))
    return digests


def digest_to_blob_relpath(digest: str) -> Path | None:
    if ":" not in digest:
        return None
    algo, hash_part = digest.split(":", 1)
    hash_part = hash_part.strip()
    if not algo or not hash_part:
        return None
    return Path(f"{algo}-{hash_part}")


def copy_blobs(digests: Set[str], source_blobs: Path, target_blobs: Path) -> tuple[int, int]:
    target_blobs.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing = 0
    for digest in sorted(digests):
        relpath = digest_to_blob_relpath(digest)
        if relpath is None:
            print(f"Skipping unsupported digest format: {digest}", file=sys.stderr)
            continue
        src = source_blobs / relpath
        dst = target_blobs / relpath
        if not src.exists():
            print(f"Missing blob for digest {digest} at {src}", file=sys.stderr)
            missing += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied, missing


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    source_manifests = base_dir / "manifests"
    source_blobs = base_dir / "blobs"

    if not source_manifests.is_dir() or not source_blobs.is_dir():
        print(
            "This script must live inside an Ollama data root containing manifests/ and blobs/",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        source_manifest_path = resolve_source_manifest(args.manifest_path, source_manifests)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    target_root = Path(args.target_root).expanduser().resolve()
    target_manifest_path = target_root / "manifests" / source_manifest_path.relative_to(source_manifests)
    target_blobs = target_root / "blobs"

    copy_manifest_payload(source_manifest_path, target_manifest_path)
    digests = gather_digests(target_manifest_path)
    copied, missing = copy_blobs(digests, source_blobs, target_blobs)

    print(f"Copied manifests into {target_manifest_path}")
    print(f"Identified {len(digests)} unique digests")
    print(f"Copied {copied} blob files to {target_blobs}")
    if missing:
        print(f"WARNING: {missing} blobs were missing in the source store", file=sys.stderr)


if __name__ == "__main__":
    main()
