#date: 2023-02-01T16:59:13Z
#url: https://api.github.com/gists/ebfe5006cc793242f183fa56841cdb1c
#owner: https://api.github.com/users/frontdevops

import builtins
import inspect
from enum import Enum


class VarType(Enum):
    DICT = 1
    LIST = 2
    OBJECT = 3
    FUNCTION = 4
    OTHER = 5


def get_var_type(expression) -> VarType:
    if isinstance(expression, dict):
        return VarType.DICT
    elif isinstance(expression, list):
        return VarType.LIST
    elif callable(expression):
        return VarType.FUNCTION
    elif hasattr(expression, '__dict__'):
        return VarType.OBJECT

    return VarType.OTHER


def var_dump(expression, level=0) -> None:
    indent = "    " * level
    var_type = get_var_type(expression)
    match var_type:
        case VarType.DICT:
            print(f"{indent}dict(size={len(expression)}) {{")
            for key, value in expression.items():
                print(f"{indent}    [{key}] =>")
                var_dump(value, level + 1)
            print(f"{indent}}}")
        case VarType.LIST:
            print(f"{indent}array(size={len(expression)}) [")
            for value in expression:
                print(f"{indent}    {value},")
            print(f"{indent}]")
        case VarType.OBJECT:
            print(f"{indent}object({expression.__class__.__name__}) {{")
            for key, value in expression.__dict__.items():
                print(f"{indent}    [{key}] =>")
                var_dump(value, level + 1)
            print(f"{indent}}}")
        case VarType.FUNCTION:
            sig = inspect.signature(expression)
            print(f"{indent}function({sig.parameters}) {{ {expression} }}")
        case VarType.OTHER:
            print(f"{indent}({type(expression).__name__}) {expression}")


# Register in global scope
builtins.var_dump = var_dump
