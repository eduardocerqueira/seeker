#date: 2025-08-29T16:58:17Z
#url: https://api.github.com/gists/2a201c46d371735612a4d28cc41f8904
#owner: https://api.github.com/users/nstarman

# astropy/units/scripts/stubgen.py
from __future__ import annotations

import importlib
import sys
import subprocess
from pathlib import Path
from typing import Iterable

def main(
    *,
    build_lib: Path,
    ...  # other stuff
) -> None:
    """
    Post-build hook that generates .pyi files for astropy.units.

    Notes
    -----
    - This function may import the built package (e.g., for inspection),
      which works because build_lib is on sys.path when the hook runs.
    """
    ...