#date: 2026-01-13T17:20:55Z
#url: https://api.github.com/gists/8aed667da7c0115983a4fd52b1638338
#owner: https://api.github.com/users/dmitrygerasimuk

#!/usr/bin/env python3

"""
map_clean.py - Normalizes MAP file from IDA for use with Soft-ICE MSYM

Transforms:
  - Hex values with 'H' suffix → pure hex
  - Fixes paragraph-aligned addresses (if needed)
  - Sanitizes symbol names
  - Enforces CRLF line endings

Usage: ida_map_to_softice.py input.map output.map
After: MSYM.EXE output.map - it will create output.msym
       Place it near your executable and launch Soft-ICE: ldr output.exe
       
"""

import re
import sys

RE_SECT_SEG = re.compile(r'^\s*Start\s+Stop\s+Length\s+Name\s+Class\s*$', re.I)
RE_SECT_PUB = re.compile(r'^\s*Address\s+Publics by Value\s*$', re.I)
RE_SEG_LINE = re.compile(r'^\s*([0-9A-Fa-f]+)H?\s+([0-9A-Fa-f]+)H?\s+([0-9A-Fa-f]+)H?\s+(\S+)\s+(\S+)\s*$')
RE_PUB_LINE = re.compile(r'^\s*([0-9A-Fa-f]{4}):([0-9A-Fa-f]{4})\s+(.+?)\s*$')

HEADER1 = " Start  Stop   Length Name               Class"
HEADER2 = ""
PUBHDR1 = " Address         Publics by Value"
PUBHDR2 = ""

def clean_sym(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[^A-Za-z0-9_\$\?\@]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    if not name:
        name = "NONAME"
    if name[0].isdigit():
        name = "_" + name
    return name

def parse_hex(s: str) -> int:
    s = s.strip()
    if s.endswith(('H', 'h')):
        s = s[:-1]
    return int(s, 16)

def looks_times16(start: int, length: int) -> bool:
    if (start & 0xF) != 0 or (length & 0xF) != 0:
        return False
    s4 = start >> 4
    l4 = length >> 4
    return (0 <= s4 <= 0xFFFF) and (1 <= l4 <= 0xFFFF)

def main(inp: str, outp: str) -> int:
    with open(inp, 'r', errors='replace') as f:
        lines = f.readlines()

    out = []
    # требование: Start Stop — строго первая строка файла
    out.append(HEADER1)
    out.append(HEADER2)

    in_seg = False
    in_pub = False
    seen_any_seg = False
    seen_any_pub = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip('\n').rstrip('\r')

        # ловим вход в секции, но заголовки сами не копируем (мы их уже вывели/выведем сами)
        if RE_SECT_SEG.match(line):
            in_seg = True
            in_pub = False
            i += 1
            continue

        if RE_SECT_PUB.match(line):
            in_pub = True
            in_seg = False
            if not seen_any_pub:
                out.append("")      # пустая строка между секциями ок
                out.append(PUBHDR1)
                out.append(PUBHDR2)
                seen_any_pub = True
            i += 1
            continue

        # сегменты
        if in_seg:
            m = RE_SEG_LINE.match(line)
            if m:
                start = parse_hex(m.group(1))
                length = parse_hex(m.group(3))
                name = m.group(4)
                klass = m.group(5)

                if looks_times16(start, length):
                    start >>= 4
                    length >>= 4

                start &= 0xFFFF
                length &= 0xFFFF
                stop = (start + length - 1) & 0xFFFF

                name2 = clean_sym(name)
                klass2 = clean_sym(klass)

                out.append(f"{start:04X} {stop:04X} {length:04X} {name2:<19} {klass2}")
                seen_any_seg = True
                i += 1
                continue

            # пустые строки просто пропускаем
            if line.strip() == "":
                i += 1
                continue

            # что-то не то — выходим из сегментной секции
            in_seg = False
            continue

        # publics
        if in_pub:
            m = RE_PUB_LINE.match(line)
            if m:
                seg = int(m.group(1), 16)
                off = int(m.group(2), 16)
                sym = clean_sym(m.group(3))
                out.append(f" {seg:04x}:{off:04x}       {sym}")
                i += 1
                continue

            # обычно хвост вида "Program entry point..." лучше выбросить
            if line.strip().lower().startswith("program entry point"):
                break

            i += 1
            continue

        i += 1

    # если в исходнике вообще не нашли сегменты, всё равно оставим только шапку (MSYM иногда это переваривает)
    # но обычно сегменты есть, просто кривые.

    # CRLF: принудительно
    with open(outp, 'w', newline="\r\n") as f:
        f.write("\r\n".join(out).rstrip() + "\r\n")

    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <in.map> <out.map>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))



    