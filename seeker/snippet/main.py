#date: 2026-03-10T17:36:18Z
#url: https://api.github.com/gists/42b235155079dd44a79e66cbcffb0060
#owner: https://api.github.com/users/mypy-play

from collections.abc import Mapping

class A: ...
class B(A): ...

def test_mapping() -> None:
    class MyMapping[K, V](Mapping[K, V]): ...

    def _0[K](arg: Mapping[K, B]) -> Mapping[K, A]: return arg
    def _1[K](arg: MyMapping[K, B]) -> MyMapping[K, A]: return arg  # ❌️