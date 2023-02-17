#date: 2023-02-17T16:52:41Z
#url: https://api.github.com/gists/70436bd1a1c4f9f3c4088cf900273996
#owner: https://api.github.com/users/wbadart

"""Tools to find free variables in an expression.

>>> import ast
>>> my_expr = ast.parse("x + (lambda x, y: x + y + z)(5, 9)").body[0].value
>>> reduce_ast(my_expr, free_vars, merge_free_name_sets, state=frozenset())
{'z', 'x'}
"""

import ast
import functools
import operator as op
import typing as t
from collections import abc

__all__ = [
    "Mapper",
    "Combiner",
    "reduce_ast",
    "free_vars",
    "merge_free_name_sets",
]


A = t.TypeVar("A")
A_co = t.TypeVar("A_co", covariant=True)
State_t = t.TypeVar("State_t")


class Mapper(t.Protocol[A_co]):
    def __call__(self, node: ast.AST, state: State_t, /) -> tuple[A_co, State_t]:
        ...


class Combiner(t.Protocol[A]):
    def __call__(self, *child_results: A) -> A:
        ...


def reduce_ast(
    node: ast.AST, mapper: Mapper[A], combiner: Combiner[A], state: t.Any
) -> A:
    my_results, new_state = mapper(node, state)
    child_results = (
        reduce_ast(child, mapper, combiner, new_state) for child in _children(node)
    )
    return combiner(my_results, *child_results)


def _children(node: ast.AST) -> abc.Iterable[ast.AST]:
    # from ast.NodeVisitor.generic_visit
    # https://github.com/python/cpython/blob/f482ade4c7887c49dfd8bba3be76f839e562608d/Lib/ast.py#L421-L429
    for _field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    yield item
        elif isinstance(value, ast.AST):
            yield value


def free_vars(
    node: ast.AST, bound_vars: abc.Set[str]
) -> tuple[abc.Set[str], abc.Set[str]]:
    if isinstance(node, ast.Name):
        return ({node.id} - bound_vars, bound_vars)
    elif isinstance(node, ast.Lambda):
        new_vars = set(_all_arg_names(node))
        return (set(), bound_vars | new_vars)
    else:
        return (set(), bound_vars)


def merge_free_name_sets(*names: abc.Set[str]) -> abc.Set[str]:
    return functools.reduce(op.or_, names)


def _all_arg_names(lambda_: ast.Lambda) -> abc.Iterable[str]:
    return (arg.arg for arg in _all_args(lambda_.args))


def _all_args(args: ast.arguments) -> abc.Iterable[ast.arg]:
    yield from args.posonlyargs
    yield from args.args
    yield from args.kwonlyargs
    if args.vararg is not None:
        yield args.vararg
    if args.kwarg is not None:
        yield args.kwarg