#date: 2023-07-18T16:58:23Z
#url: https://api.github.com/gists/f3d66b02b1d88fccd72b43e319c5e1c2
#owner: https://api.github.com/users/mypy-play

from pathlib import Path
from typing import Optional


def get_build_dir(cluster: str, cwd: Optional[Path] = None) -> Path:
    if cluster == "local":
        relpath = Path("foo")
    else:
        relpath = Path("bar")

    if cwd is not None:
        return (cwd / relpath).absolute()
    else:
        return relpath.absolute()