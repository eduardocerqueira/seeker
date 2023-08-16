#date: 2023-08-16T16:42:14Z
#url: https://api.github.com/gists/90cd186173e46eb55603fda12006bd49
#owner: https://api.github.com/users/enzo-santos

import os
import functools

import typing


Node: typing.TypeAlias = None | dict[str, "Node"]
Tree: typing.TypeAlias = dict[str, Node]


def parse(tree: Tree, *, prefix: str = "output"):
    try:
        os.makedirs(prefix)
    except FileExistsError:
        pass

    fnames: list[str] = []
    for key, node in tree.items():
        if node is None:
            with open(os.path.join(prefix, f"{key}.html"), "w+", encoding="utf-8") as f:
                print(f"<p>{key}</p>", file=f)

        else:
            parse(node, prefix=os.path.join(prefix, key))

        fnames.append(f"{key}.html" if node is None else f"{key}/")

    with open(os.path.join(prefix, "index.html"), "w+", encoding="utf-8") as f:
        printf = functools.partial(print, file=f)

        title = f"Index of {prefix.removeprefix('output') or os.path.sep}"

        printf("<!DOCTYPE HTML>")
        printf('<html lang="pt">')
        printf("<head>")
        printf('<meta charset="utf-8">')
        printf(f"<title>{title}</title>")
        printf("</head>")
        printf("<body>")
        printf(f"<h1>{title}</h1>")
        printf("<hr>")
        printf("<ul>")
        for fname in fnames:
            printf(f'<li><a href="{fname}">{fname}</a></li>')
        printf("</ul>")
        printf("<hr>")
        printf("</body>")
        printf("</html>")


def main() -> None:
    # This is equivalent to the following,
    # completely arbitrary directory structure:
    # |_ assets/
    # |   |_ alfa.html
    # |   |_ bravo.html
    # |   |_ charlie.html
    # |   |_ delta.html
    # |_ project/
    # |   |_ lib/
    # |   |   |_ i.html
    # |   |   |_ ii.html
    # |   |   |_ iii.html
    # |   |   |_ iv.html
    # |   |_ i/
    # |   |   |_ 00.html
    # |   |   |_ 01.html
    # |   |   |_ 02.html
    # |   |_ ...
    # |   |_ iv/
    # |   |   |_ 00.html
    # |   |   |_ 01.html
    # |   |   |_ 02.html
    # |   |_ README.html
    # |_ README.html
    data: Tree = {
        "assets": dict.fromkeys(("alfa", "bravo", "charlie", "delta")),
        "project": {
            "lib": dict.fromkeys(("i", "ii", "iii", "iv")),
            "i": dict.fromkeys(map("{:02d}".format, range(3))),
            "ii": dict.fromkeys(map("{:02d}".format, range(3))),
            "iii": dict.fromkeys(map("{:02d}".format, range(3))),
            "iv": dict.fromkeys(map("{:02d}".format, range(3))),
            "README": None,
        },
        "README": None,
    }

    parse(data)


if __name__ == "__main__":
    main()
