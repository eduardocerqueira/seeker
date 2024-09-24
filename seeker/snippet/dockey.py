#date: 2024-09-24T16:55:35Z
#url: https://api.github.com/gists/753b353b457c45faf2cefbfcf4bf23b7
#owner: https://api.github.com/users/tdwiser

from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class DocKey:
    key: str
    __doc__: str
    def __repr__(self):
        return repr(self.key)
    def __hash__(self):
        return hash(self.key)
    def __eq__(self, value):
        return self.key == value

sample_dict = {"key_a": 1, "key_b": 2}
doc_keys = [DocKey("key_a", "this is key A"), DocKey("key_b", "this is key B")]

for key in doc_keys:
    print(f"{key=}, {key.__doc__=}, {sample_dict[key]=}")

help(doc_keys[0])
