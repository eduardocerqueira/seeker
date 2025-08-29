#date: 2025-08-29T16:58:17Z
#url: https://api.github.com/gists/2a201c46d371735612a4d28cc41f8904
#owner: https://api.github.com/users/nstarman

# SPDX-License-Identifier: BSD-3-Clause
# extension_helpers/hooks.py
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


def mypy_stubgen(
    *,
    build_lib: Path,
    dist,
    pkg: str,
    modules: str | None = None,
    out: str = "auto",
    extra: str | None = None,
) -> None:
    """
    Run 'python -m mypy.stubgen' after build_ext and place .pyi into build_lib.

    Parameters
    ----------
    pkg : str
        Top-level package to stub, e.g., 'yourpkg'.
    modules : str, optional
        Space- or comma-separated list of specific modules to stub,
        e.g. "yourpkg.sub yourpkg._fast". If omitted, stubgen receives -p <pkg>.
    out : {"auto", "<path>"}
        Where to write stubs. "auto" means <build_lib>/<pkg>.
    extra : str, optional
        Extra arguments to pass verbatim to stubgen (space-separated).

    Notes
    -----
    - Requires 'mypy' in the build environment (PEP 517 build backend env).
    - We *import nothing* here; stubgen may import modules itself, and because
      we injected 'build_lib' on sys.path, those imports resolve to the built
      artifacts (C/Cython extensions included).
    """
    if out == "auto":
        out_dir = build_lib / pkg.replace(".", "/")
    else:
        out_dir = Path(out)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the command
    cmd: list[str] = [sys.executable, "-m", "mypy.stubgen"]
    if modules:
        # allow comma or space separated
        items: Sequence[str] = [
            m for m in modules.replace(",", " ").split() if m.strip()
        ]
        for m in items:
            cmd += ["-m", m]
    else:
        cmd += ["-p", pkg]

    cmd += ["-o", str(out_dir)]
    if extra:
        cmd += extra.split()

    # Run stubgen
    subprocess.check_call(cmd)