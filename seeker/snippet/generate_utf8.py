#date: 2024-11-01T16:45:25Z
#url: https://api.github.com/gists/b06ca3d976d26fa011897bd1603ea044
#owner: https://api.github.com/users/rendello

import sys
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Entry:
    a: str
    b: str
    a_len: int
    b_len: int
    delta: int
    a_char_count: int
    b_char_count: int
    delta_char_count: int


def sort_entries(l: List[Entry]) -> List[Entry]:
    """ Sorted by size delta, then alphabetically. """
    return sorted(l, key=
        lambda p: (-(p.delta_char_count), -(p.delta), p.a))


def create_entry_map() -> dict[str, list[Entry]]:
    entry_map = {}
    for i in range(sys.maxunicode + 1):
        a = chr(i)

        for (case, b) in (('uppercasing', a.upper()), ('lowercasing', a.lower())):
            attributes = [case]

            try:
                a_len = len(a.encode("utf8"))
                b_len = len(b.encode("utf8"))
            except UnicodeEncodeError:
                continue

            if a_len == b_len:
                continue

            delta = b_len - a_len

            a_char_count = len(a)
            b_char_count = len(b)
            delta_char_count = b_char_count - a_char_count

            if a_len < b_len:
                attributes.append('expands')
            elif a_len > b_len:
                attributes.append('contracts')

            if b_char_count > 1:
                attributes.append('multi_char')

            key = "_".join(attributes)
            value = Entry(a, b, a_len, b_len, delta, a_char_count, b_char_count, delta_char_count)

            if key not in entry_map:
                entry_map[key] = [value]
            else:
                entry_map[key].append(value)

    return entry_map


def entry_map_to_string(entry_map: Dict[str, List[Entry]]) -> str:
    buffer = (
        f'''// =======================================================================\n'''
        f'''//! Automatically generated using `task generate-utf8-case-data`.\n//!\n'''
        f'''//! Unicode characters that behave oddly when the case is changed, for use\n'''
        f'''//! with property tests.\n'''
        f'''// =======================================================================\n\n'''
    )
    for key, unsorted_entries in sorted(list(entry_map.items())):
        entries = sort_entries(unsorted_entries)

        buffer += f'pub const {key.upper()}: [&str; {len(entries)}] = [\n'
        for e in entries:

            ds = ""
            if e.delta_char_count != 0:
                ds = f", {e.delta_char_count:+} chars"

            buffer += f'    "{e.a}",\t/* {e.b}\t({e.a_len}->{e.b_len}), {e.delta:+} bytes{ds} */\n'
        buffer += "];\n\n"
    return buffer.strip()


def generate_utf8_case_data():
    return entry_map_to_string(create_entry_map())