#date: 2023-10-26T16:39:14Z
#url: https://api.github.com/gists/811a3d24f7832ace1e63efd86bf8a793
#owner: https://api.github.com/users/mypy-play

from typing import Mapping, MutableMapping, Sequence, MutableSequence, Union

JsonValue = Union["JsonObject", "JsonArray", str, int, float, bool, None]
JsonObject = Mapping[str, JsonValue]
JsonArray = Sequence[JsonValue]

# MutableJsonValue = Union["JsonObject", "JsonArray", str, int, float, bool, None]
MutableJsonObject = MutableMapping[str, JsonValue]
# MutableJsonArray = MutableSequence[MutableJsonValue]


j: MutableJsonObject

j["foo"] = "bar"

reveal_type(j["foo2"])
assert isinstance(j["foo2"], dict)
reveal_type(j["foo2"])
j["foo2"]["bar"] = "baz"