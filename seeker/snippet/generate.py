#date: 2025-08-22T17:01:38Z
#url: https://api.github.com/gists/d84baa0919ce9dd023e661aec35a4b8e
#owner: https://api.github.com/users/hnbdr

"""
Generate `.pyi` files for GObject instrospect libraries
such as Gtk and Gdk.

Python version 3.10

Requires `gobject-introspection` package to be
installed at `/usr/lib/gobject-introspection/`.

Does not support (yet)
- `@staticmethod`s
- `__init__`s
- `@classmethod`s

For a more complete generator see
https://github.com/santiagocezar/gengir
"""

from collections import defaultdict
from pathlib import Path
import sys
from typing import Callable, Iterable, Type, TypeVar
from multiprocessing import Pool

sys.path.append("/usr/lib/gobject-introspection/")

from giscanner.girparser import GIRParser
from giscanner import ast
import textwrap
import warnings

T = TypeVar("T")
K = TypeVar("K")


def param_to_docstr(p: ast.Parameter):
    direction = "param" if p.direction == "in" else "return"
    return f":{direction} {p.name}: {p.doc}"


class OmitMethod(Exception):
    """
    Do not include the method from which this exception is raised.
    """


def gtype_to_pytype(t: ast.Type) -> str:
    match t:
        case ast.TYPE_STRING | \
             ast.TYPE_FILENAME | \
             ast.TYPE_UNICHAR:
            return "str"
        case ast.TYPE_BOOLEAN:
            return "bool"
        case ast.TYPE_DOUBLE | \
             ast.TYPE_FLOAT:
            return "float"
        case ast.TYPE_INT8 | \
             ast.TYPE_UINT8 | \
             ast.TYPE_INT16 | \
             ast.TYPE_UINT16 | \
             ast.TYPE_INT32 | \
             ast.TYPE_UINT32 | \
             ast.TYPE_INT64 | \
             ast.TYPE_UINT64 | \
             ast.TYPE_SHORT | \
             ast.TYPE_USHORT | \
             ast.TYPE_INT | \
             ast.TYPE_UINT | \
             ast.TYPE_LONG | \
             ast.TYPE_ULONG | \
             ast.TYPE_SIZE | \
             ast.TYPE_SSIZE |\
             ast.Type(target_fundamental="guint8"):
            return "int"
        case ast.TYPE_ANY:
            return "Any"
        case ast.Type(target_giname=str()):
            return t.target_giname
        case ast.Type(resolved="GType"):
            return "GObject.GType"
        case ast.Type(resolved=("<array>" | "<list>")):
            return f"list[{gtype_to_pytype(t.element_type)}]"
        case ast.Map():
            return (
                f"dict[{gtype_to_pytype(t.key_type)}, {gtype_to_pytype(t.value_type)}]"
            )
        case ast.Varargs() | ast.TYPE_VALIST:
            raise OmitMethod()
        case ast.TYPE_NONE:
            return "None"
        case _:
            warnings.warn(f'Unrecognized type {t!r}')
            return 'Any'


def handle_arg(p: ast.Parameter) -> str:
    # Escape keyword param arguments
    assert p.direction == 'in'
    name = p.name
    if name in {'def'}:
        name += '_'
    arg = f"{name}: {gtype_to_pytype(p.type)}"
    if p.optional:
        arg += ' = ...'
    return arg


def _handle_func(f: ast.Function) -> tuple[str, str, str]:
    """
    Returns args str, return type and definition
    """
    definition = " ..."
    doc = ""
    if f.doc:
        doc = f.doc
    if f.parameters:
        if doc:
            doc += "\n\n"
        doc += "\n".join(param_to_docstr(p) for p in f.parameters)
    if doc:
        definition = textwrap.indent(f'\n"""\n{doc}\n"""\n...', "  ")
    args = ', '.join(handle_arg(p) for p in f.parameters if p.direction == 'in')
    ret = gtype_to_pytype(f.retval.type)
    extra_ret =', '.join(gtype_to_pytype(p.type) for p in f.parameters if p.direction == 'out')
    if extra_ret:
        ret = f'tuple[{ret}, {extra_ret}]'
    return (args, ret, definition)

def handle_func(f: ast.Function) -> str:
    try:
        (args, ret, definition) = _handle_func(f)
        return f"""\
def {f.name}({args}) -> {ret}:{definition}
"""
    except OmitMethod: return ""

def handle_method(f: ast.Function) -> str:
    try:
        (args, ret, definition) = _handle_func(f)
        return f"""\
def {f.name}(self, {args}) -> {ret}:{definition}
"""
    except OmitMethod: return ""


def handle_class(cls: ast.Class) -> str:
    # TODO: Add properties, attributes, signals
    # static_methods and virtual_methods.
    doc = ""
    if cls.doc:
        doc = textwrap.indent(f'"""\n{cls.doc}\n"""\n', "  ")

    methods = textwrap.indent("\n\n".join(handle_method(f) for f in cls.methods), "  ")
    parent_str = ""
    if cls.parent_type:
        parent_str = f"({cls.parent_type.resolved})"
    if not doc and not methods:
        doc = "  ..."
    return f"""\
class {cls.name}{parent_str}:
{doc}{methods}
"""


def grouped_by(
    items: Iterable[T],
    key: Callable[[T], K],
) -> dict[K, list[T]]:
    d = defaultdict(list)
    for i in items:
        d[key(i)].append(i)
    return d


def handle_constant(c: ast.Constant) -> str:
    typ = ""
    match c.value_type:
        case ast.TYPE_INT8 | \
             ast.TYPE_UINT8 | \
             ast.TYPE_INT16 | \
             ast.TYPE_UINT16 | \
             ast.TYPE_INT32 | \
             ast.TYPE_UINT32 | \
             ast.TYPE_INT64 | \
             ast.TYPE_UINT64 | \
             ast.TYPE_SHORT | \
             ast.TYPE_USHORT | \
             ast.TYPE_INT | \
             ast.TYPE_UINT | \
             ast.TYPE_LONG | \
             ast.TYPE_ULONG | \
             ast.TYPE_SIZE | \
             ast.TYPE_SSIZE |\
             ast.TYPE_DOUBLE | \
             ast.TYPE_FLOAT | \
             ast.TYPE_CHAR:
            val = c.value
        case ast.TYPE_STRING:
            val = f'"{c.value}"'
        case ast.TYPE_BOOLEAN:
            val = 'True' if c.value == 'true' else 'False'
        case ast.Type(target_giname=str()):
            typ = f': {c.value_type.target_giname}'
            val = c.value
        case _:
            raise TypeError(f"Unsupported constant type. {c.value_type}, {c}")
    doc = textwrap.indent(c.doc or "", "# ", lambda l: True)
    if doc:
        doc = f"\n{doc}\n"
    return f"{doc}{c.name}{typ} = {val}"


def handle_enum(e: ast.Enum) -> str:
    values = textwrap.indent(
        "\n".join(f"{m.name.upper()} = {m.value}" for m in e.members), "  "
    )
    doc = ""
    if e.doc:
        doc = textwrap.indent(f'"""\n{e.doc}\n"""\n', "  ")
    if not values and not doc:
        values = "  ..."
    return f"""\
class {e.name}(GObject.Enum):
{doc}{values}
"""


def gen_namespace(ns: ast.Namespace) -> str:

    by_type: dict[Type[ast.Node], list[ast.Node]] = grouped_by(
        ns.values(), key=lambda x: type(x)
    )
    constants = "\n".join(handle_constant(c) for c in by_type.pop(ast.Constant, ()))
    enums = "\n\n".join(handle_enum(e) for e in by_type.pop(ast.Enum, ()))
    classes = "\n\n".join(handle_class(e) for e in by_type.pop(ast.Class, ()))
    functions = '\n\n'.join(handle_func(f) for f in by_type.pop(ast.Function, ()))

    return f"""\
{constants}

{enums}

{classes}

{functions}
"""

def recurse_deps(*incs: ast.Include) -> set[ast.Include]:
    deps = set()
    def _rec(inc: ast.Include):
        if inc in deps: return
        deps.add(inc)
        for subdep in modules[(inc.name, inc.version)][1]:
            _rec(subdep)
    for inc in incs: _rec(inc)
    return deps


def generate_stub(ns: ast.Namespace, inc: set[ast.Include]):
    inc = recurse_deps(*inc)
    namespace = gen_namespace(ns)
    gi_deps = textwrap.indent('\n'.join((
         # Implicit deps of multiple pkgs
        'GObject,  # 2.0',
        'Gio,  # 2.0',
        'GLib,  # 2.0',
        # The pkg itself
        f'{ns.name},  # {ns.version}',
        # And the actual dependencies
        *(f"{i.name},  # {i.version}" for i in inc)
    )), '  ')
    return f"""\
'''
{ns.name} v{ns.version}
'''
from typing import Any
from gi.repository import (
{gi_deps}
)

__version__ = "{ns.version}"

{namespace}
"""

modules: dict[tuple[str, str], tuple[ast.Namespace, set[ast.Include]]] = {}

Path('./stubs/gi/repository').mkdir(parents=True, exist_ok=True)

def parse_gir(p: Path) -> tuple[tuple[str, str], tuple[ast.Namespace, set[ast.Include]]]:
    name, _, version = p.name.removesuffix(".gir").rpartition("-")
    g = GIRParser()
    g.parse(str(p))
    assert g._namespace is not None
    assert name == g._namespace.name
    assert version == g._namespace.version
    return (name, version), (g._namespace, g._includes)

with Pool(8) as pool:
    modules = dict(pool.map(parse_gir, (
        p for p in Path("/usr/share/gir-1.0/").iterdir()
        if p.suffix == '.gir'
    )))

for (name, ver), (ns, inc) in modules.items():
   Path(f'./stubs/gi/repository/{name}.pyi').write_text(
       generate_stub(ns, inc)
   )
