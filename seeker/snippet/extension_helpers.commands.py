#date: 2025-08-29T16:58:17Z
#url: https://api.github.com/gists/2a201c46d371735612a4d28cc41f8904
#owner: https://api.github.com/users/nstarman

# SPDX-License-Identifier: BSD-3-Clause
# extension_helpers/commands.py
from __future__ import annotations

import contextlib
import importlib
import os
import shlex
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

from setuptools.command.build_py import build_py as _build_py

# Python 3.11+: stdlib tomllib; otherwise use tomli (declare in build-system.requires)
try:
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    tomllib = None  # will try tomli dynamically

WELL_KNOWN_HOOKS = {
    "mypy-stubgen": "extension_helpers.hooks:mypy_stubgen",
    # "pybind11-stubgen": "extension_helpers.hooks:pybind11_stubgen",
}


def _iter_declared_hooks(dist) -> Iterable[str]:
    """
    Yield hook specifications from configuration.

    Precedence:
      1) pyproject.toml -> [tool.extension-helpers].post_build_hooks (list of strings)
      2) env var EH_POST_BUILD_HOOKS = "hook1 args; hook2 args"
    """
    # 1) pyproject.toml
    for spec in _hooks_from_pyproject():
        yield spec
    else:
        # 2) environment fallback (optional)
        env = os.environ.get("EH_POST_BUILD_HOOKS", "").strip()
        if env:
            for part in env.split(";"):
                part = part.strip()
                if part:
                    yield part


def _hooks_from_pyproject() -> Iterable[str]:
    """Read hooks from pyproject.toml: [tool.extension-helpers].post_build_hooks."""
    # We assume build is run from the project root (PEP 517 backends do this).
    pp = Path("pyproject.toml")
    if not pp.is_file():
        return ()

    # Load TOML
    data = None
    if tomllib is not None:
        with pp.open("rb") as f:
            data = tomllib.load(f)
    else:  # fallback to tomli if available
        with contextlib.suppress(Exception):
            import tomli  # type: ignore

            with pp.open("rb") as f:
                data = tomli.load(f)

    if not isinstance(data, dict):
        return ()
    tool = data.get("tool")
    if not isinstance(tool, dict):
        return ()
    ext = tool.get("extension-helpers") or tool.get("extension_helpers")
    if not isinstance(ext, dict):
        return ()

    hooks = ext.get("post_build_hooks")
    if hooks is None:
        return ()
    if isinstance(hooks, str):
        # allow single string (split by newline/semicolon for convenience)
        for candidate in _split_hooks_string(hooks):
            yield candidate
        return ()
    if isinstance(hooks, list):
        for item in hooks:
            if isinstance(item, str) and item.strip():
                yield item.strip()
    # else: ignore invalid shapes
    return ()


def _split_hooks_string(s: str) -> Iterable[str]:
    # Support "hook a=b; other c=d" and multi-line strings
    for part in s.replace("\n", ";").split(";"):
        part = part.strip()
        if part:
            yield part


def _load_callable(spec: str) -> Tuple[Callable[..., None], dict]:
    """
    Parse a hook spec and return (callable, kwargs).

    Forms:
      1) well-known:  "mypy-stubgen pkg=yourpkg out=auto"
      2) dotted path: "pkg.module:func key=value"
    """
    parts = shlex.split(spec)
    if not parts:
        raise ValueError("Empty hook specification")

    head, *arg_parts = parts

    if head in WELL_KNOWN_HOOKS:
        target = WELL_KNOWN_HOOKS[head]
    else:
        if ":" not in head:
            raise ValueError(
                f"Hook '{head}' is neither a well-known name nor 'pkg.module:func'"
            )
        target = head

    mod_name, func_name = target.split(":", 1)
    func = getattr(importlib.import_module(mod_name), func_name)

    kwargs = {}
    for ap in arg_parts:
        if "=" not in ap:
            raise ValueError(f"Hook argument '{ap}' must be key=value")
        k, v = ap.split("=", 1)
        kwargs[k] = v

    return func, kwargs


class BuildPyWithPostBuild(_build_py):
    """Run build_ext first, execute post-build hooks, then normal build_py."""

    description = "build pure Python modules (with post-build-ext hooks)"

    def run(self):
        # 1) Ensure compiled artifacts are present
        self.run_command("build_ext")

        # 2) Make build_lib importable for hooks
        build_lib = Path(self.build_lib)
        sys.path.insert(0, str(build_lib))

        try:
            hooks_ran: List[str] = []
            for spec in _iter_declared_hooks(self.distribution):
                func, kwargs = _load_callable(spec)
                func(build_lib=build_lib, dist=self.distribution, **kwargs)
                hooks_ran.append(spec)

            if hooks_ran:
                self.announce(
                    "extension-helpers: executed post-build hooks:\n  - "
                    + "\n  - ".join(hooks_ran),
                    level=2,
                )
        finally:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(build_lib))

        # 3) Continue as usual
        super().run()