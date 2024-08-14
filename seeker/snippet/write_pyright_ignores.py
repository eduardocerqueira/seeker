#date: 2024-08-14T18:58:29Z
#url: https://api.github.com/gists/1e5b370ecff3f7f1e15d27ce6c29c044
#owner: https://api.github.com/users/kevin-hanselman

import sys
import json
import itertools
import tempfile
import shutil


def index_file_diagnostics(diagnostics):
    out = {}
    for diag in diagnostics:
        # pyright's JSON output either has a bug or intentionally
        # uses zero-based indexing.
        # See: https://github.com/microsoft/pyright/issues/8760
        key = diag["range"]["start"]["line"] + 1
        if key in out:
            out[key].add(diag["rule"])
        else:
            out[key] = {diag["rule"]}
    return out


def write_ignores(*, file_in, file_out, indexed_diags):
    for line_num, line in enumerate(file_in, start=1):
        rules = indexed_diags.get(line_num, None)
        line = line.rstrip()
        if rules:
            line += f"  # pyright: ignore[{','.join(rules)}]"
        file_out.write(f"{line}\n")


if __name__ == "__main__":
    pyright_report = json.load(sys.stdin)

    grouped_diagnostics = itertools.groupby(
        pyright_report["generalDiagnostics"], key=lambda d: d["file"]
    )

    for path, file_diagnostics in grouped_diagnostics:
        indexed_diags = index_file_diagnostics(file_diagnostics)

        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix="add_pyright_ignores",
            delete_on_close=False,
        ) as file_out:
            with open(path, "r") as file_in:
                write_ignores(file_in=file_in, file_out=file_out, indexed_diags=indexed_diags)
            file_out.close()
            print(f"{file_out.name} -> {path}")
            shutil.move(src=file_out.name, dst=path)