#date: 2025-03-21T17:00:52Z
#url: https://api.github.com/gists/05a26912120062b4dcb5e8af00f1af56
#owner: https://api.github.com/users/niltonfrederico

import functools

from typing import Callable
from typing import Iterable
from typing import Self

from django.utils.text import slugify


def parametrize(fieldnames: str | Iterable[str], values: Iterable[Iterable]) -> list[dict]:
    if isinstance(fieldnames, str):
        fieldnames = [field.strip() for field in fieldnames.split(",")]

    def _parametrize(test_method: Callable):
        test_method.__parametrize__ = fieldnames, values
        return test_method

    return _parametrize


def _make_test(method: Callable, params: dict) -> Callable:
    @functools.wraps(method)
    def test_wrapper(self):
        return method(self, **params)

    return test_wrapper


def _get_parametrize_values(values: Iterable) -> Iterable:
    if values == "":
        values = [""]
    else:
        values = list(values) if not isinstance(values, (list, tuple)) else values

    return values


class ParametrizedTestCaseMeta(type):
    def __new__(cls: Self, name: str, bases: tuple, attrs: dict):
        # We force attrs.items to be a list to avoid RuntimeError: dictionary changed size during iteration
        for method_name, method_callable in list(attrs.items()):
            if hasattr(method_callable, "__parametrize__"):
                params, values = method_callable.__parametrize__

                # We need to remove the original test method from the class
                del attrs[method_name]

                # We need to create a new test method for each set of parameters
                for position, value in enumerate(values):
                    test_slug = slugify(value).replace("-", "_")
                    value = _get_parametrize_values(value)

                    test_name = f"{method_name}_{position}_{test_slug}"

                    test_method = _make_test(method_callable, dict(zip(params, value)))
                    attrs[test_name] = test_method

        return super().__new__(cls, name, bases, attrs)
