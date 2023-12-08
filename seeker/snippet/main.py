#date: 2023-12-08T16:37:51Z
#url: https://api.github.com/gists/de938ce4ee905e67af27f05b235f212f
#owner: https://api.github.com/users/mypy-play

from typing_extensions import Self  # for 3.10-compatibility

class Foo:
    def return_self(self) -> Self:
        return self

class SubclassOfFoo(Foo):
    """I am a subclass."""

reveal_type(Foo().return_self())  # Revealed type is "Foo"
reveal_type(SubclassOfFoo().return_self())  # Revealed type is "SubclassOfFoo"