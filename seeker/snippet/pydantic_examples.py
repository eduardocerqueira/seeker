#date: 2025-09-04T16:59:32Z
#url: https://api.github.com/gists/f7b5eaf4a986e03574da8e008fb9dfd3
#owner: https://api.github.com/users/thejchap

# /// script
# dependencies = ["pydantic"]
# ///

from typing import Self

from pydantic import BaseModel, computed_field


class A(BaseModel):
    field_a: str
    field_b: str


class B(A):
    field_c: str

    @classmethod
    def from_a(cls, a: A) -> Self:
        """
        need to explicitly pass each field into constructor - i think this is nicer/more explicit than the splat operator
        and gives type errors in addition to runtime errors if anything is wrong
        """
        return cls(
            field_c=f"{a.field_a} {a.field_b}!",
            field_a=a.field_a,
            field_b=a.field_b,
        )


class C(A):
    @computed_field
    @property
    def field_c(self) -> str:
        """
        another option is to lean into computed fields
        you can construct a `C` from an `A` and nothing else via `from_attributes=True`,
        then the calculations for `C`'s new fields are computed based on the stuff from `A`
        """
        return f"{self.field_a} {self.field_b}!"


def main():
    a = A(field_a="hello", field_b="world")
    b = B.from_a(a)
    print(b.field_c)  # => "hello world!"

    b = C.model_validate(a, from_attributes=True)
    print(b.field_c)  # => "hello world!"


if __name__ == "__main__":
    main()
