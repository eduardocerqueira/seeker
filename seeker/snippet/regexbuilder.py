#date: 2022-02-16T17:11:22Z
#url: https://api.github.com/gists/9b24acac7e61b9ff97d4296a0de04e1c
#owner: https://api.github.com/users/kageurufu

import re

ALLOWED_TYPES = {"comment", "reply", "unsub", "sub"}


class RegexBuilder:
    def __init__(self, *parts):
        self._parts = []
        if parts:
            self.add(*parts)

    def __repr__(self):
        return "<RegexBuilder '%s'>" % self

    def __str__(self):
        return "".join(str(s) for s in self._parts)

    def build(self, flags=0):
        return re.compile(str(self), flags)

    def __add__(self, other):
        self.add(other)

    def add(self, *other):
        for part in other:
            self._parts.append(part)
        return self

    def optional(self, *other):
        return self.add(r"(?:").add(*other).add(r")?")

    def capture(self, name, *other):
        return self.add(r"(?P<" + name + r">").add(*other).add(r")")

    def ignore(self, *other):
        return self.add(r"(?:").add(*other).add(r")")


_signature_with_type_regex_builder = (
    RegexBuilder()
    .optional(r"[\w]+\.")  # Optional signature "type"
    .add(r"[a-f0-9]+")  # Signed value
    .optional(r"\.[a-f0-9]+")  # Optional timestamp
    .add(r"\.[a-f0-9]+")  # Signature
)

signed_email_regex = (
    RegexBuilder()
    .capture("type", r"|".join(ALLOWED_TYPES))  # Type of incoming message
    .add(r"[\+-]")  # Separator
    .capture("key", _signature_with_type_regex_builder)  # Signature match
    .ignore(r"@.+")  # Ignore domain
    .build(re.IGNORECASE)
)


def parse_inbound_email_address(email):
    if match := signed_email_regex.match(email):
        return match.groupdict()
