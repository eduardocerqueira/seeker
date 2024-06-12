#date: 2024-06-12T16:48:16Z
#url: https://api.github.com/gists/85e55553f85c410a1b856a93dce77208
#owner: https://api.github.com/users/julian-klode

import os
import sys

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import NoteSection


def open_file(filename):
    elffile = ELFFile.load_from_path(filename)
    print(filename, "elf")
    for sect in elffile.iter_sections():
        if sect.name != ".note.gnu.build-id":
            continue
        for note in sect.iter_notes():
            build_id = note["n_desc"]
            debug_file = f"/usr/lib/debug/.build-id/{build_id[:2]}/{build_id[2:]}.debug"
            if not os.path.exists(debug_file):
                print(filename, "no-dbgsym", build_id)
            if debug_file != filename:
                return ELFFile.load_from_path(debug_file)

    print(filename, "using-binary")
    return elffile


def process_file(filename):
    elffile = open_file(filename)
    dwarfinfo = elffile.get_dwarf_info(follow_links=True)
    for CU in dwarfinfo.iter_CUs():
        die = CU.get_top_DIE()
        print(
            filename,
            "unit",
            die.attributes["DW_AT_name"].value.decode("utf-8"),
            "produced-by",
            die.attributes["DW_AT_producer"].value.decode("utf-8"),
        )


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        process_file(filename)
