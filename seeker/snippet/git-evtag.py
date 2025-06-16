#date: 2025-06-16T17:06:37Z
#url: https://api.github.com/gists/038d60b3fb1f3913d93de0bb49102aba
#owner: https://api.github.com/users/bbhtt

#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-2.0-or-later
# Original: https://github.com/cgwalters/git-evtag/blob/main/src/git-evtag-compute-py

import argparse
import hashlib
import subprocess
import types
from pathlib import Path
from typing import IO, Self

GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG": "''",
}


class ChecksumProcessor:
    def __init__(self) -> None:
        self.stats = {
            "commit": 0,
            "tree": 0,
            "blob": 0,
            "commitbytes": 0,
            "treebytes": 0,
            "blobbytes": 0,
        }
        self.csum = hashlib.sha512()

    def update(self, kind: str, data: bytes) -> int:
        data_len = len(data)
        self.csum.update(data)
        self.stats[kind + "bytes"] += data_len
        return data_len

    def increment(self, kind: str) -> None:
        self.stats[kind] += 1

    def print_digest(self) -> None:
        print(f"Tree checksum: {self.csum.hexdigest()}")  # noqa: T201


class GitBatchProcessor:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self._process: None | subprocess.Popen[bytes] = None
        self._stdin: None | IO[bytes] = None
        self._stdout: None | IO[bytes] = None

    def __enter__(self) -> Self:
        self._process = subprocess.Popen(
            ["git", "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=self.repo,
            env=GIT_ENV,
        )
        if not (self._process.stdin and self._process.stdout):
            raise RuntimeError("Failed to open subprocess streams")
        self._stdin = self._process.stdin
        self._stdout = self._process.stdout
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self._stdin:
            self._stdin.close()
        if self._process:
            self._process.wait()
            if self._process.returncode != 0:
                raise subprocess.CalledProcessError(
                    self._process.returncode, "git cat-file --batch"
                )

    def get_object(self, obj_id: str) -> tuple[str, int, bytes]:
        if not (self._stdin and self._stdout):
            raise RuntimeError("Batch process not initialized")

        self._stdin.write(obj_id.encode("ascii") + b"\n")
        self._stdin.flush()

        header = self._stdout.readline().decode("ascii").strip()
        if " missing" in header:
            raise ValueError(f"Object {obj_id} not found")

        parts = header.split(None, 2)
        if len(parts) != 3:
            raise ValueError(f"Malformed header: {header}")

        obj_id_returned, obj_type, str_len = parts
        obj_len = int(str_len)

        content = self._stdout.read(obj_len)
        if len(content) != obj_len:
            raise ValueError(f"Expected {obj_len} bytes, got {len(content)}")

        self._stdout.read(1)

        return obj_type, obj_len, content


class GitProcessor:
    def __init__(self, repo: Path, checksum: ChecksumProcessor) -> None:
        self.repo = repo
        self.checksum = checksum

    def checksum_object(self, batch_proc: GitBatchProcessor, obj_id: str) -> None | str:
        if not obj_id:
            raise ValueError("Object ID must not be None")

        obj_type, obj_len, content = batch_proc.get_object(obj_id)

        buf = f"{obj_type} {obj_len}\0".encode("ascii")
        self.checksum.update(obj_type, buf)
        self.checksum.increment(obj_type)

        tree_obj_id: str | None = None

        if obj_type == "commit":
            lines = content.decode("ascii").split("\n")
            if lines and lines[0].startswith("tree "):
                tree_obj_id = lines[0].split(None, 1)[1].strip()
            else:
                raise ValueError("Malformed commit object, expected 'tree <sha>' line")

        self.checksum.update(obj_type, content)

        return tree_obj_id

    def checksum_tree(
        self, batch_proc: GitBatchProcessor, path: Path, obj_id: str
    ) -> None:
        self.checksum_object(batch_proc, obj_id)

        ret = subprocess.Popen(
            ["git", "ls-tree", obj_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            cwd=self.repo,
            env=GIT_ENV,
        )

        if not ret.stdout:
            raise RuntimeError("Failed to open stdout for ls-tree")

        for line in ret.stdout:
            mode, obj_type, subid, fname = line.decode("ascii").split(None, 3)
            fname = fname.strip()

            if obj_type == "blob":
                self.checksum_object(batch_proc, subid)
            elif obj_type == "tree":
                self.checksum_tree(batch_proc, path / fname, subid)
            elif obj_type == "commit":
                subrepo = self.repo / path / fname
                subproc = GitProcessor(subrepo, self.checksum)
                with GitBatchProcessor(subrepo) as sub_batch_proc:
                    subproc.checksum_repo(sub_batch_proc, subid, path / fname)
            else:
                raise ValueError(f"Unknown object type: {obj_type}")

        ret.wait()
        if ret.returncode != 0:
            raise subprocess.CalledProcessError(ret.returncode, "git ls-tree")

    def checksum_repo(
        self, batch_proc: GitBatchProcessor, obj_id: str, path: Path = Path(".")
    ) -> None:
        tree_id = self.checksum_object(batch_proc, obj_id)
        if tree_id:
            self.checksum_tree(batch_proc, path, tree_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Checksum a Git repository tree")
    parser.add_argument("--rev", default="HEAD", help="Git revision (default: HEAD)")
    parser.add_argument(
        "--repo", default=".", help="Path to the Git repository (default: current dir)"
    )
    args = parser.parse_args()

    checksum = ChecksumProcessor()
    repo = Path(args.repo).resolve()
    processor = GitProcessor(repo, checksum)

    with GitBatchProcessor(repo) as batch_proc:
        processor.checksum_repo(batch_proc, args.rev, repo)

    checksum.print_digest()


if __name__ == "__main__":
    main()
