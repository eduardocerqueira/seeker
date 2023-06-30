#date: 2023-06-30T16:41:33Z
#url: https://api.github.com/gists/bc4fea6b1545dbb4825341d00683c6ef
#owner: https://api.github.com/users/mypy-play

import typing as t

T = t.TypeVar("T")
_T = t.TypeVar("_T")

class IsType(t.Generic[T]):
    @classmethod
    def check(cls: T, val: _T) -> t.TypeGuard[T]:
        return True


feature_by_name: dict[str, t.Any] = {
    "a": "allo", 
    "b": ["hello", "world"],
}

named_features: list[tuple[str, t.TypeAlias]] = [
    ("a", str),
    ("b", list[str]),
]

t.reveal_type(feature_by_name)
t.reveal_type(named_features)

for name, feature_type in named_features:
    value = feature_by_name[name]
    print(feature_type)
    #check_result = IsType[feature_type].check(value)
    #print(f"{check_result}, {name}, {feature_type}, {value}, {type(value)}")
        
print("a")


