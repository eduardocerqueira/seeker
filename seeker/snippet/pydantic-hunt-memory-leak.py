#date: 2022-05-13T17:08:37Z
#url: https://api.github.com/gists/a5509ed60ddf0dd5bfbc692385c80b85
#owner: https://api.github.com/users/samuelcolvin

import os
from typing import TypeVar, Generic

import psutil
import pytest

from pydantic.generics import GenericModel

process = psutil.Process(os.getpid())

last = 0
while True:
    # pytest.main(['tests/test_abc.py', '-qq'])
    # pytest.main(['tests/test_aliases.py', '-qq'])
    # pytest.main(['tests/test_annotated.py', '-qq'])
    # pytest.main(['tests/test_annotated_types.py', '-qq'])
    # pytest.main(['tests/test_assert_in_validators.py', '-qq'])
    # pytest.main(['tests/test_callable.py', '-qq'])
    # pytest.main(['tests/test_color.py', '-qq'])
    # pytest.main(['tests/test_construction.py', '-qq'])
    # pytest.main(['tests/test_create_model.py', '-qq'])
    # pytest.main(['tests/test_dataclasses.py', '-qq'])

    # pytest.main(['tests/test_datetime_parse.py', '-qq'])
    # pytest.main(['tests/test_decorator.py', '-qq'])
    # pytest.main(['tests/test_discrimated_union.py', '-qq'])

    # pytest.main(['tests/test_edge_cases.py', '-qq'])
    # pytest.main(['tests/test_errors.py', '-qq'])
    # pytest.main(['tests/test_forward_ref.py', '-qq'])

    # pytest.main(['tests/test_generics.py', '-qq'])

    # pytest.main(['tests/test_json.py', '-qq'])
    # pytest.main(['tests/test_main.py', '-qq'])
    # pytest.main(['tests/test_model_signature.py', '-qq'])
    # pytest.main(['tests/test_networks_ipaddress.py', '-qq'])
    # pytest.main(['tests/test_networks.py', '-qq'])
    # pytest.main(['tests/test_orm_mode.py', '-qq'])
    # pytest.main(['tests/test_parse.py', '-qq'])
    # pytest.main(['tests/test_private_attributes.py', '-qq'])
    # pytest.main(['tests/test_schema.py', '-qq'])
    # pytest.main(['tests/test_settings.py', '-qq'])
    # pytest.main(['tests/test_tools.py', '-qq'])
    # pytest.main(['tests/test_types_payment_card_number.py', '-qq'])
    # pytest.main(['tests/test_types.py', '-qq'])
    # pytest.main(['tests/test_typing.py', '-qq'])
    # pytest.main(['tests/test_utils.py', '-qq'])
    # pytest.main(['tests/test_validators_dataclass.py', '-qq'])
    # pytest.main(['tests/test_validators.py', '-qq'])
    # pytest.main(['tests/test_version.py', '-qq'])

    T = TypeVar('T')
    S = TypeVar('S')

    class A(GenericModel, Generic[T, S]):
        ...

    class B(A[str, T], Generic[T]):
        ...

    assert B[int].__name__ == 'B[int]'
    # assert issubclass(B[int], A[str, int])
    # assert not issubclass(B[str], A[str, int])


    mem = process.memory_info().rss
    mb = 1024 * 1024
    print(f'{mem / mb:8.2f}MB {(mem - last) / mb:+8.2f}MB | {"━" * int(mem / 10_000_000)}')
    last = mem
