#date: 2025-10-09T16:57:55Z
#url: https://api.github.com/gists/5139276df420591427a365070fae73e8
#owner: https://api.github.com/users/ScottVinay

from pathlib import Path

def find_git_root(start: Optional[Path]=None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    for p in (start, *start.parents):
        if (p / ".git").exists() or (p / "pyproject.toml").exists():
            return p
    else:
        raise FileNotFoundError("No git repo root found")