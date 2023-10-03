#date: 2023-10-03T16:48:04Z
#url: https://api.github.com/gists/219f4d7bbfcf50b1d92f627dd0587106
#owner: https://api.github.com/users/mypy-play

from pathlib import Path
from typing import ParamSpec, Protocol, cast, Callable, Concatenate


PCompressor = ParamSpec("PCompressor")


# class Compressor(Protocol[PCompressor]):
#     def __call__(
#         self, input_fn: Path, /, *args: PCompressor.args, **kwargs: PCompressor.kwargs
#     ) -> None:
#         ...
Compressor = Callable[Concatenate[Path, PCompressor], None]


def compressor_a(input_fn: Path, setting_a: str):
    ...


def compressor_b(input_fn: Path, setting_b: int):
    ...


def compress(
    input_fn: Path,
    compressor: Compressor[PCompressor],
    /,
    *compr_args: PCompressor.args,
    **compr_kwargs: PCompressor.kwargs,
):
    compressor(input_fn, *compr_args, **compr_kwargs)


p = Path("foo")
compress(p, compressor_a, setting_a="foo")   # should pass
compress(p, compressor_b, setting_b=42)      # should pass
compress(p, compressor_b, does_not_exist=1)  # should fail
compress(p, compressor_b, setting_b="foo")   # should fail: incompatible int, expected str

