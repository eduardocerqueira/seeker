#date: 2025-10-06T16:57:49Z
#url: https://api.github.com/gists/cf1af5c9efc659a0b1336d92aa05257b
#owner: https://api.github.com/users/yochem

import inspect
import typing
from textwrap import dedent
from pathlib import Path


def public_property(obj):
    if public_method(obj):
        return False
    return isinstance(obj, property)


def public_method(obj):
    if not (inspect.isfunction(obj) or inspect.ismethod(obj)):
        return False
    if obj.__name__.startswith("_") and obj.__name__ != "__init__":
        return False
    return True


def codeblock(code):
    return f"```python\n{code}\n```"


def heading(name, level=1):
    return f"{'#' * level} {name}"


def gh_permalink(file, obj):
    file = Path(file).relative_to(Path(".").absolute())
    lines, start = inspect.getsourcelines(obj)
    end = start + len(lines)
    # TODO: get commit from git
    commit = "5d02d3acddb5fea513415f593c972dca7418e806"
    url = f"https://github.com/yochem/py-spidev/blob/{commit}/{file}#L{start}-L{end}"
    return f"[Full source]({url})"


def details(content):
    return f"""<details><summary>Signature</summary>\n\n{content}\n\n</details>\n"""


def printdoc(obj):
    doc = inspect.getdoc(obj)
    if doc:
        print(doc)
    print()


def document_class(cls):
    print(heading(cls.__name__, 2))
    printdoc(cls)

    file = inspect.getsourcefile(cls)

    print(heading("Properties", 3))
    for name, m in inspect.getmembers(cls, public_property):
        ptype = m.fget.__annotations__["return"]
        print(heading(f"`{name}`: {ptype}", 4))
        printdoc(m)

    print()

    print(heading("Methods", 3))
    for name, m in inspect.getmembers(cls, public_method):
        signature = inspect.signature(m)

        # method parameters with 'self' removed
        params = dict(signature.parameters)
        del params["self"]

        # simple signature
        print(heading(f"`{name}({', '.join(params)})`", 4))

        # full (type-hints included) signature
        clean_signature = signature.format(max_width=80).replace("'", "")
        text = f'def {name}{clean_signature}'
        print(details(codeblock(text) + "\n" + gh_permalink(file, m)))

        printdoc(m)


if __name__ == "__main__":
    import spidev

    printdoc(spidev)

    document_class(spidev.SpiDev)

    with open('spi-numbering.md') as f:
        print(f.read())