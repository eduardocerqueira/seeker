#date: 2023-09-01T17:05:23Z
#url: https://api.github.com/gists/b79f176ca1b7855d3f849bb4c2028fcf
#owner: https://api.github.com/users/iamahuman

#!/usr/bin/python3
#
# Configure metalink URL query parameters for system DNF repositories
# Copyright (C) 2023  Jinoh Kang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing import Dict, FrozenSet, Iterable, List, Optional, OrderedDict, Tuple
import sys
from urllib.parse import urlsplit, urlunsplit
import collections
import logging
import argparse

import dnf  # type: ignore
import libdnf.conf  # type: ignore

logger = logging.getLogger("s-ml-p")

DNF_METALINK_KEY = "metalink"

RESERVED_MIRRORLIST_KEYS = frozenset(
    (
        "arch",
        "repo",
    )
)


def set_urlencoded_qs_values(
    query: str, key: str, values: Iterable[Optional[str]]
) -> str:
    if query:
        items = query.split("&")
    else:
        items = []

    i = 0

    for value in values:
        pair = key + "=" + value if value is not None else key
        while True:
            if i == len(items):
                items.append(pair)
                i += 1
                break
            elif items[i].split("=", 1)[0] == key:
                items[i] = pair
                i += 1
                break

            i += 1

    while i < len(items):
        if items[i].split("=", 1)[0] == key:
            del items[i]
        else:
            i += 1

    return "&".join(items)


assert set_urlencoded_qs_values("", "foo", []) == ""
assert set_urlencoded_qs_values("", "foo", ["1"]) == "foo=1"
assert set_urlencoded_qs_values("", "foo", ["1", "2"]) == "foo=1&foo=2"
assert set_urlencoded_qs_values("", "foo", [None]) == "foo"
assert set_urlencoded_qs_values("", "foo", ["%$ +"]) == "foo=%$ +"
assert set_urlencoded_qs_values("novalue", "foo", []) == "novalue"
assert set_urlencoded_qs_values("novalue", "foo", ["1"]) == "novalue&foo=1"
assert set_urlencoded_qs_values("novalue", "foo", ["1", "2"]) == "novalue&foo=1&foo=2"
assert set_urlencoded_qs_values("novalue", "foo", [None]) == "novalue&foo"
assert set_urlencoded_qs_values("bar=0", "foo", []) == "bar=0"
assert set_urlencoded_qs_values("bar=0", "foo", ["1"]) == "bar=0&foo=1"
assert set_urlencoded_qs_values("bar=0", "foo", ["1", "2"]) == "bar=0&foo=1&foo=2"
assert set_urlencoded_qs_values("bar=0", "foo", [None]) == "bar=0&foo"
assert set_urlencoded_qs_values("bar=2&baz=3", "bar", ["y"]) == "bar=y&baz=3"
assert set_urlencoded_qs_values("bar=2&baz=3", "bar", ["y", "z"]) == "bar=y&baz=3&bar=z"
assert set_urlencoded_qs_values("bar=2&bar=3", "bar", ["y"]) == "bar=y"
assert set_urlencoded_qs_values("bar=2&bar=3", "bar", ["y", "z"]) == "bar=y&bar=z"
assert (
    set_urlencoded_qs_values("foo=1&bar=2&baz=4&bar=3", "bar", [None, "z"])
    == "foo=1&bar&baz=4&bar=z"
)


def set_url_query_param_values(
    url: str, key: str, values: Iterable[Optional[str]]
) -> str:
    scheme: str
    netloc: str
    path: str
    query: str
    fragment: str

    scheme, netloc, path, query, fragment = urlsplit(url)

    new_query = set_urlencoded_qs_values(query, key, values)

    return urlunsplit((scheme, netloc, path, new_query, fragment))


assert (
    set_url_query_param_values("https://example.com/path?foo=0#fragment", "foo", [])
    == "https://example.com/path#fragment"
)
assert (
    set_url_query_param_values("https://example.com/path?foo=0#fragment", "foo", ["1"])
    == "https://example.com/path?foo=1#fragment"
)
assert (
    set_url_query_param_values("https://example.com/path?foo=0#fragment", "foo", [None])
    == "https://example.com/path?foo#fragment"
)


def get_repofile_sections_from_dnf(
    dnf_base: dnf.Base, enabled_only: bool, disabled_only: bool
) -> List[Tuple[str, List[str]]]:
    result: Dict[str, List[str]] = {}
    for repo in dnf_base.repos.all():
        if enabled_only and not repo.enabled:
            continue

        if disabled_only and repo.enabled:
            continue

        result.setdefault(repo.repofile, []).append(repo.id)
    return sorted(result.items())


def set_repofile_metalink_parameters(
    filename: str,
    section_ids: Iterable[str],
    substitutions: Optional[Dict[str, str]],
    overrides: OrderedDict[str, List[Optional[str]]],
    dry_run: bool,
) -> int:
    parser = libdnf.conf.ConfigParser()

    logger.debug("Config %r reading", filename)
    parser.read(filename)

    num_changes = 0

    for section_id in section_ids:
        if substitutions is not None and not parser.hasSection(section_id):
            for raw_section in parser.getData():
                subst_section = libdnf.conf.ConfigParser.substitute(
                    raw_section, substitutions
                )
                if subst_section == section_id:
                    logger.debug(
                        "Translating section ID %r to %r", section_id, raw_section
                    )
                    section_id = raw_section

        if not parser.hasSection(section_id):
            logger.warning("Cannot find section %r", section_id)
        elif not parser.hasOption(section_id, DNF_METALINK_KEY):
            logger.info(
                "Cannot find option %r in section %r", DNF_METALINK_KEY, section_id
            )
        else:
            logger.debug(
                "Processing option %r in section %r", DNF_METALINK_KEY, section_id
            )

            old_metalink: str = parser.getValue(section_id, DNF_METALINK_KEY)
            new_metalink: str = old_metalink
            for key, values in overrides.items():
                new_metalink = set_url_query_param_values(new_metalink, key, values)

            if old_metalink != new_metalink:
                logger.info(
                    "Option %r [%r] %r changed", filename, section_id, DNF_METALINK_KEY
                )
                logger.info(
                    "- [%r] %r old value: %r",
                    section_id,
                    DNF_METALINK_KEY,
                    old_metalink,
                )
                logger.info(
                    "- [%r] %r new value: %r",
                    section_id,
                    DNF_METALINK_KEY,
                    new_metalink,
                )

                num_changes += 1
                parser.setValue(section_id, DNF_METALINK_KEY, new_metalink)
            else:
                logger.debug(
                    "Option %r [%r] %r unchanged",
                    filename,
                    section_id,
                    DNF_METALINK_KEY,
                )

    if num_changes:
        if not dry_run:
            logger.info(
                "Config %r %d option(s) changed, writing", filename, num_changes
            )
            parser.write(filename, False)
        else:
            logger.info(
                "Config %r %d option(s) changed, not writing (dry run)",
                filename,
                num_changes,
            )
    else:
        logger.info("Config %r is unchanged", filename)

    return num_changes


class ArgParseError(Exception):
    pass


def parse_overrides(
    unset_keys: Iterable[str],
    parameters: Iterable[str],
) -> OrderedDict[str, List[Optional[str]]]:
    overrides: OrderedDict[str, List[Optional[str]]] = collections.OrderedDict()

    for item in parameters:
        for pair in item.split("&"):
            if "#" in pair:
                raise ArgParseError(
                    "Invalid character '#' in parameter {!r}".format(pair)
                )

            key, sep, value = pair.partition("=")
            overrides.setdefault(key, []).append(value if sep else None)

    for unset_key in unset_keys:
        for invalid_char in "&=":
            if invalid_char in unset_key:
                raise ArgParseError(
                    "Invalid character {!r} in unset key {!r}".format(
                        invalid_char, unset_key
                    )
                )

        if unset_key in overrides:
            raise ArgParseError(
                "Cannot simultaneously set and unset parameter {!r}".format(unset_key)
            )

        overrides[unset_key] = []

    reserved_set_keys = RESERVED_MIRRORLIST_KEYS.intersection(overrides.keys())
    if reserved_set_keys:
        raise ArgParseError(
            "Cannot modify reserved key(s): "
            + ", ".join(map(repr, sorted(reserved_set_keys)))
        )

    return overrides


def main(argv: List[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Configure metalink URL query parameters for system DNF repositories",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        required=False,
        default=0,
        help="increase verbosity",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="count",
        required=False,
        default=0,
        help="decrease verbosity",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        required=False,
        default=False,
        help="don't actually write anything (use with -v)",
    )
    enabled_group = parser.add_mutually_exclusive_group()
    enabled_group.add_argument(
        "-e",
        "--enabled-only",
        action="store_true",
        required=False,
        default=False,
        help="Only modify enabled repositories",
    )
    enabled_group.add_argument(
        "-d",
        "--disabled-only",
        action="store_true",
        required=False,
        default=False,
        help="Only modify disabled repositories",
    )
    parser.add_argument(
        "-u",
        "--unset",
        action="append",
        required=False,
        default=[],
        help="mirrorlist URL query parameters to unset",
    )
    parser.add_argument(
        "parameters",
        nargs="*",
        help="mirrorlist URL query parameters to override (e.g., country=XX,YY,ZZ)",
    )
    args = parser.parse_args(argv)

    log_levels = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
        logging.NOTSET,
    ]
    log_level = log_levels[
        max(0, min(len(log_levels) - 1, 2 + args.verbose - args.quiet), 0)
    ]
    logging.basicConfig(level=log_level)

    try:
        overrides = parse_overrides(
            unset_keys=args.unset,
            parameters=args.parameters,
        )
    except ArgParseError as ex:
        parser.error(ex.args[0])

    if not overrides:
        parser.error("Nothing to do")

    base = dnf.Base()
    base.read_all_repos()

    for repofile, sections in get_repofile_sections_from_dnf(
        base, enabled_only=args.enabled_only, disabled_only=args.disabled_only
    ):
        set_repofile_metalink_parameters(
            filename=repofile,
            section_ids=sections,
            substitutions=base.conf.substitutions,
            overrides=overrides,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
