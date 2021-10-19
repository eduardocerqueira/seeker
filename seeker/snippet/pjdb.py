#date: 2021-10-19T17:14:09Z
#url: https://api.github.com/gists/0ea3b2b78c1a0f20851fd377bea51f19
#owner: https://api.github.com/users/conqp

#! /usr/bin/env python3
#
# pjdb.py - JSON-ify pacman databases.
#
# (C) 2021 Richard Neumann <mail at richard dash neumann period de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""JSON-ify pacman database."""

from json import dumps
from pathlib import Path
from subprocess import CompletedProcess, run
from sys import argv
from tempfile import TemporaryDirectory
from typing import Iterator, NamedTuple, Union


DESCFILE = 'desc'
INT_KEYS = {'%BUILDDATE%', '%CSIZE%', '%ISIZE%'}
LIST_KEYS = {'%DEPENDS%', '%MAKEDEPENDS%', '%OPTDEPENDS%'}
LIST_SEP = '\n'
ITEM_SEP = '\n\n'
KEY_VALUE_SEP = '\n'
TAR = '/usr/bin/tar'
JSON = Union[str, int, list[str]]


class KeyValuePair(NamedTuple):
    """A key-value pair of package meta data."""

    key: str
    value: JSON


class PackageDescription(NamedTuple):
    """Desription of a package."""

    key: str
    value: dict[str, JSON]


def _tar_xf(tarfile: Path, directory: Path) -> CompletedProcess:
    """Workaround for missing zstd support in Python's tar library."""

    return run([TAR, 'xf', str(tarfile), '-C', str(directory)], check=True)


def kv_from_str(text: str) -> Iterator[KeyValuePair]:
    """Yields key / value pairs from a string."""

    for item in text.split(ITEM_SEP):
        if not item.strip():
            continue

        key, value = item.split(KEY_VALUE_SEP, maxsplit=1)

        if key in INT_KEYS:
            value = int(value)
        elif key in LIST_KEYS:
            value = value.split(LIST_SEP)

        yield KeyValuePair(key.replace('%', '').lower(), value)


def kv_from_file(filename: Path) -> Iterator[KeyValuePair]:
    """Yields key / value pairs from a file."""

    with filename.open('r') as file:
        yield from kv_from_str(file.read())


def pkg_from_dir(dirname: Path) -> PackageDescription:
    """Reads package information from a package directory."""

    return PackageDescription(
        dirname.name, dict(kv_from_file(dirname / DESCFILE)))


def pkgs_from_dir(dirname: Path) -> Iterator[PackageDescription]:
    """Yields package descriptions."""

    for pkgdir in dirname.iterdir():
        yield pkg_from_dir(pkgdir)


def pkgs_from_db(filename: Path) -> Iterator[PackageDescription]:
    """Yields package descriptions from a database file."""

    with TemporaryDirectory() as tmpd:
        tmpd = Path(tmpd)
        _tar_xf(filename, tmpd)
        yield from pkgs_from_dir(tmpd)


def main():
    """Test the above stuff."""

    databases = {}

    for database in argv[1:]:
        databases[database] = dict(pkgs_from_db(database))

    print(dumps(databases, indent=2))


if __name__ == '__main__':
    main()