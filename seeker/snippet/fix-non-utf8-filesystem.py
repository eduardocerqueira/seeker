#date: 2023-01-31T16:57:06Z
#url: https://api.github.com/gists/7b3a82b20b9a20ade0b132b25791ef08
#owner: https://api.github.com/users/ldotlopez

import os
import sys


def is_utf_8(s):
    try:
        return s.encode("utf-8").decode("utf-8") == s
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def get_utf8_error(s):
    try:
        s.encode("utf-8").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        return str(e)


def simple_fix(s):
    tbl = {
        "\udc80": "Ç", "\udc81": "ü", "\udc82": "é", "\udc85": "à",
        "\udc87": "ç", "\udc8a": "é", "\udc8b": "ï", "\udc95": "ó",
        "\udca0": "à", "\udca1": "í", "\udca1": "í", "\udca2": "ó",
        "\udca2": "ó", "\udca3": "ú", "\udca4": "ñ", "\udca6": "ª",
        "\udca7": "ª", "\udcb7": "·", "\udcc9": "É", "\udce0": "à",
        "\udce7": "ç", "\udce9": "è", "\udced": "í", "\udcf1": "í",
        "\udcf3": "ó", "\udcfa": "·"
    }

    fixed = "".join([tbl.get(x, x) for x in s])

    if not is_utf_8(fixed):
        raise UnicodeError(fixed)

    return fixed


def process(where):
    for root, dirs, files in os.walk(where, topdown=False):
        for entry in files + dirs:
            origpath = root + "/" + entry

            if not is_utf_8(entry):
                fixedpath = root + "/" + simple_fix(entry)

                if os.path.exists(fixedpath):
                    raise ValueError(f"{fixedpath}: already exists")

                yield (origpath, fixedpath)


for src, dst in process(sys.argv[0]):
    os.rename(src, dst)
    # print(f"{src!r} => {dst!r}")
