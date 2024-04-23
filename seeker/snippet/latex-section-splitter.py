#date: 2024-04-23T16:46:33Z
#url: https://api.github.com/gists/deaa4366fd013b486f42a867ef25d9d2
#owner: https://api.github.com/users/snwnde

# The MIT License (MIT)

# Copyright (c) 2024 Sen-wen Deng

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import pathlib
from TexSoup import TexSoup  # type: ignore


def parse_path(path: str) -> pathlib.Path:
    path_obj = pathlib.Path(path)
    if path_obj.is_absolute():
        return path_obj
    return pathlib.Path.cwd() / path_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Separate a flat LaTeX file into sections"
    )
    parser.add_argument(
        "file",
        type=str,
        help="""Path to the LaTeX file to separate.
        Either absolute or relative to the current working directory.""",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="""Path to the directory where the separated sections will be saved.
        Either absolute or relative to the current working directory.""",
        default="sections",
    )
    parser.add_argument(
        "--output-main",
        type=str,
        help="""Path to the main file where the sections will be included.
        Either absolute or relative to the current working directory.""",
        default="main.tex",
    )
    args = parser.parse_args()

    file_desc = parse_path(args.file)
    dir_path = parse_path(args.output_dir)
    main_desc = parse_path(args.output_main)

    with open(file_desc) as f:
        soup = TexSoup(f, tolerance=0)

    sections = {
        str(section.string): section.position
        for section in soup.document.find_all(name=["section", "section*"])
    }
    # Put the bibliography in the main file
    bib_position = soup.find_all(name="bibliography")[0].position
    end_position = bib_position if bib_position else soup.document[-1].position
    # Gather the positions of the split points
    split_positions = list(sections.values()) + [end_position]
    main_up = str(soup)[: split_positions[0]]
    main_down = "\t" + str(soup)[split_positions[-1] :]
    main_includes = []
    for idx, sec in enumerate(sections.keys()):
        dir_path.mkdir(exist_ok=True, parents=True)
        name = sec.lower()
        slice_ = slice(split_positions[idx], split_positions[idx + 1])
        section_content = str(soup)[slice_].rstrip() + "\n"

        with open(dir_path / f"{name}.tex", "w") as f:
            f.write(section_content)

        prefix = "" if idx == 0 else "\t"
        main_includes.append(
            f"{prefix}\\input{{{dir_path.relative_to(file_desc.parent)}/{name}.tex}}\n"
        )

    main_includes_ = "".join(main_includes)
    new_main = main_up + main_includes_ + main_down

    with open(main_desc, "w") as f:
        f.write(new_main)
