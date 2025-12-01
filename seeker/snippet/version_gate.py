#date: 2025-12-01T17:02:08Z
#url: https://api.github.com/gists/8f19cae8dba2bcf1b436ddc9a0700c92
#owner: https://api.github.com/users/vjayajv

"""
Version Gate Pre-Run Modifier for Robot Framework

This modifier filters Robot Framework tests based on version tags:

- minver:X.Y.Z      Run for versions >= X.Y.Z
- maxver:X.Y.Z      Run for versions <= X.Y.Z
- onlyver:<expr>    Run for versions matching a version expression
                    (e.g. >=8.4,<9.0,!=8.6.0)

Version filtering is applied only to tests that explicitly opt in
(via tags such as upgrade, pre-upgrade, post-upgrade, sanity).

The product version is read from the PRODUCT_VERSION environment variable.
"""

import operator
import os
import re
from functools import total_ordering
from typing import Optional

from robot.api import SuiteVisitor


@total_ordering
class Version:
    """Semantic version parser and comparator."""

    def __init__(self, version_string: str):
        self.original = version_string.strip()

        # Remove leading 'v' if present
        clean_version = self.original.lstrip("vV")

        # Strip build metadata or prerelease suffix
        if "-" in clean_version:
            clean_version = clean_version.split("-", 1)[0]
        if "+" in clean_version:
            clean_version = clean_version.split("+", 1)[0]

        parts = clean_version.split(".")

        try:
            self.major = int(parts[0]) if len(parts) > 0 else 0
            self.minor = int(parts[1]) if len(parts) > 1 else 0
            self.patch = int(parts[2]) if len(parts) > 2 else 0
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid version format: {version_string}") from e

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __repr__(self) -> str:
        return f"Version({self.original})"

    def __eq__(self, other) -> bool:
        return (
            self.major,
            self.minor,
            self.patch,
        ) == (
            other.major,
            other.minor,
            other.patch,
        )

    def __lt__(self, other) -> bool:
        return (
            self.major,
            self.minor,
            self.patch,
        ) < (
            other.major,
            other.minor,
            other.patch,
        )


class VersionConstraint:
    """Parser and evaluator for version expressions."""

    OPERATORS = {
        ">=": operator.ge,
        "<=": operator.le,
        ">": operator.gt,
        "<": operator.lt,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def __init__(self, expression: str):
        self.original = expression.strip()
        self.constraints = []

        for chunk in expression.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue

            match = re.match(r"(>=|<=|>|<|==|!=)\s*(.+)", chunk)
            if not match:
                raise ValueError(f"Invalid constraint expression: {chunk}")

            op, version_str = match.groups()
            version = Version(version_str)
            self.constraints.append((op, version))

    def matches(self, version: Version) -> bool:
        for op, target in self.constraints:
            if not self.OPERATORS[op](version, target):
                return False
        return True

    def __str__(self) -> str:
        return self.original


class VersionGateVisitor(SuiteVisitor):
    """
    Robot Framework pre-run modifier that filters tests based on version tags.

    Usage:
        robot --prerunmodifier version_gate.VersionGateVisitor tests/
    """

    PRODUCT_VERSION_ENV = "PRODUCT_VERSION"

    # Tags that opt tests into version gating
    GATED_TAGS = {
        "upgrade",
        "pre-upgrade",
        "post-upgrade",
        "pre_upgrade",
        "post_upgrade",
        "sanity",
    }

    MINVER_PREFIX = "minver:"
    MAXVER_PREFIX = "maxver:"
    ONLYVER_PREFIX = "onlyver:"

    def __init__(self, version: Optional[str] = None):
        version_str = version or os.environ.get(self.PRODUCT_VERSION_ENV)

        if not version_str:
            print(
                "WARNING: PRODUCT_VERSION is not set. "
                "Version gating is disabled; all tests will run."
            )
            self.enabled = False
            self.version = None
            return

        try:
            self.version = Version(version_str)
            self.enabled = True
            print(f"Version Gate enabled for PRODUCT_VERSION={self.version}")
        except ValueError as exc:
            print(f"ERROR: Invalid PRODUCT_VERSION '{version_str}': {exc}")
            print("Version gating is disabled; all tests will run.")
            self.enabled = False
            self.version = None

        self.removed_tests = []
        self.total_gated_tests = 0

    def start_suite(self, suite):
        if not self.enabled:
            return

        suite.tests = [
            test for test in suite.tests if self._should_run_test(test)
        ]

    def end_suite(self, suite):
        if not self.enabled or suite.parent is not None:
            return

        if not self.removed_tests:
            return

        print("\n" + "=" * 70)
        print(f"Version Gate Summary (PRODUCT_VERSION={self.version})")
        print("=" * 70)
        print(f"Total gated tests evaluated: {self.total_gated_tests}")
        print(f"Tests removed: {len(self.removed_tests)}")

        print("\nRemoved tests:")
        for name, reason in self.removed_tests[:20]:
            print(f"  - {name}: {reason}")

        if len(self.removed_tests) > 20:
            print(f"  ... and {len(self.removed_tests) - 20} more")

        print("=" * 70 + "\n")

    def _should_run_test(self, test) -> bool:
        tags = {str(tag).lower() for tag in test.tags}

        if not (tags & self.GATED_TAGS):
            return True

        self.total_gated_tests += 1

        minver = self._extract_tag(tags, self.MINVER_PREFIX)
        maxver = self._extract_tag(tags, self.MAXVER_PREFIX)
        onlyver = self._extract_tag(tags, self.ONLYVER_PREFIX)

        if not (minver or maxver or onlyver):
            return True

        try:
            if onlyver:
                return self._check_onlyver(test, onlyver)
            return self._check_minmax(test, minver, maxver)
        except ValueError as exc:
            print(
                f"WARNING: Invalid version tag in test '{test.name}': {exc}. "
                "Test will run."
            )
            return True

    def _check_onlyver(self, test, expression: str) -> bool:
        constraint = VersionConstraint(expression)
        if constraint.matches(self.version):
            return True

        self._record_removal(
            test, f"onlyver:{expression} excludes {self.version}"
        )
        return False

    def _check_minmax(
        self,
        test,
        minver: Optional[str],
        maxver: Optional[str],
    ) -> bool:
        if minver and self.version < Version(minver):
            self._record_removal(
                test, f"{self.version} < minver:{minver}"
            )
            return False

        if maxver and self.version > Version(maxver):
            self._record_removal(
                test, f"{self.version} > maxver:{maxver}"
            )
            return False

        return True

    @staticmethod
    def _extract_tag(tags: set, prefix: str) -> Optional[str]:
        for tag in tags:
            if tag.startswith(prefix):
                return tag[len(prefix) :]
        return None

    def _record_removal(self, test, reason: str):
        self.removed_tests.append((test.name, reason))


def pre_run_modify(suite):
    """
    Convenience entry point.

    Usage:
        robot --prerunmodifier version_gate:pre_run_modify tests/
    """
    suite.visit(VersionGateVisitor())


__all__ = [
    "VersionGateVisitor",
    "Version",
    "VersionConstraint",
    "pre_run_modify",
]
