#date: 2025-11-18T17:15:28Z
#url: https://api.github.com/gists/e026ed1a245c46c967055f15a7b38ee1
#owner: https://api.github.com/users/sureshjoshi

# /// script
# dependencies = [
#   "pytest==9.0.1",
# ]
# requires-python = ">=3.11"
# ///

import json
import pytest
import subprocess
import sys

from pathlib import Path
from string import Template
from typing import Final

SCIENCE_VERSION = "v0.15.1"
SCIENCE_BINARY: Final[str] = "science-fat-macos-aarch64"

LIFT_TEMPLATE: Final[Template] = Template("""
[lift]
name = "foo-bin"
load_dotenv = true
platforms = [
    "macos-aarch64",
]

[lift.scie_jump]
version = "$jump_version"

[[lift.interpreters]]
id = "cpython"
provider = "PythonBuildStandalone"
release = "20251031"
version = "3.11.14"
flavor = "install_only_stripped"
lazy = true

[[lift.files]]
name = "main.py"

[[lift.commands]]
exe = "#{cpython:python}"
args = [
    "{main.py}",
]
""")

MAIN_PY: Final[str] = """
import os

for k,v in sorted(os.environ.items()):
    print (f"{k}={v}")
"""

DOTENV: Final[str] = f"""
SCIE_ZZZ=123
ZZZ=456
"""


@pytest.fixture(scope="session")
def science_bin_path(tmp_path_factory) -> Path:
    science_dir = tmp_path_factory.mktemp("tools")
    subprocess.run(
        f"gh release download --repo a-scie/lift {SCIENCE_VERSION} --clobber --dir {science_dir} --pattern '{SCIENCE_BINARY}'",
        shell=True,
        check=True,
    )
    science_bin = science_dir / SCIENCE_BINARY
    subprocess.run(
        f"gh attestation verify --repo a-scie/lift {science_bin}",
        shell=True,
        check=True,
    )
    subprocess.run(f"chmod +x {science_bin}", shell=True, check=True)
    return science_bin


def test_science_bin_uses_jump_180(science_bin_path: Path):
    scie_json = _inspect_scie(science_bin_path)
    assert scie_json["jump"] == {"version": "1.8.0", "size": 1849240}


@pytest.mark.parametrize(
    "jump_version, expected",
    [
        ("1.8.0", {"version": "1.8.0", "size": 1849240}),
        ("1.7.0", {"version": "1.7.0", "size": 1799448}),
    ],
)
def test_jump_version_is_present(
    jump_version: str, expected: dict, tmp_path: Path, science_bin_path: Path
):
    lift_path = tmp_path / "lift.toml"
    lift_path.write_text(
        LIFT_TEMPLATE.substitute(jump_version=jump_version, encoding="utf-8")
    )

    main_py = tmp_path / "main.py"
    main_py.write_text(MAIN_PY)

    _run_science_build(science_bin_path, lift_path, working_dir=tmp_path)
    scie_json = _inspect_scie(tmp_path / "foo-bin")
    assert scie_json["jump"] == expected


@pytest.mark.parametrize("jump_version", ["1.6.1", "1.7.0", "1.8.0"])
def test_dotenv_works(jump_version: str, tmp_path: Path, science_bin_path: Path):
    lift_path = tmp_path / "lift.toml"
    lift_path.write_text(
        LIFT_TEMPLATE.substitute(jump_version=jump_version, encoding="utf-8")
    )

    main_py = tmp_path / "main.py"
    main_py.write_text(MAIN_PY)

    _run_science_build(science_bin_path, lift_path, working_dir=tmp_path)
    stdout, _ = _run_scie(tmp_path / "foo-bin", working_dir=tmp_path)
    # Stripping out my unrelated, for easier diffing
    env_lines = [
        line for line in stdout.splitlines() if "SCIE" in line or "ZZZ" in line
    ]
    assert "SCIE_ZZZ=123" not in env_lines
    assert "ZZZ=456" not in env_lines

    dotenv = tmp_path / ".env"
    dotenv.write_text(DOTENV)

    stdout, _ = _run_scie(tmp_path / "foo-bin", working_dir=tmp_path)
    env_lines = [
        line for line in stdout.splitlines() if "SCIE" in line or "ZZZ" in line
    ]
    assert "SCIE_ZZZ=123" in env_lines
    assert "ZZZ=456" in env_lines


def _run_science_build(science_path: Path, lift_path: Path, working_dir: Path):
    subprocess.run(
        f"{science_path} lift build {lift_path}",
        shell=True,
        check=True,
        cwd=working_dir,
    )


def _inspect_scie(scie_path: Path) -> dict:
    result = subprocess.run(
        f"{scie_path}",
        shell=True,
        check=True,
        env={"SCIE": "inspect"},
        capture_output=True,
    )
    return json.loads(result.stdout)["scie"]


def _run_scie(scie_path: Path, working_dir: Path) -> tuple[str, str]:
    result = subprocess.run(
        f"{scie_path}",
        shell=True,
        check=True,
        cwd=working_dir,
        capture_output=True,
        text=True,
    )
    return result.stdout, result.stderr


if __name__ == "__main__":
    print(__file__)
    sys.exit(pytest.main([__file__, "-vv", f"--config-file={__file__}"]))
