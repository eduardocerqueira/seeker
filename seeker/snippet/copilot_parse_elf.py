#date: 2021-11-22T17:14:41Z
#url: https://api.github.com/gists/fb59d357e7941d91c550abced6698bdc
#owner: https://api.github.com/users/moyix

#!/usr/bin/env python3

import sys
import struct
from pprint import pprint

# Parser for ELF executables

def parse_elf_header(f):
    # Parse ELF header
    f.seek(0)
    magic = f.read(4)
    if magic != b'\x7fELF':
        raise Exception('Not an ELF file')
    f.seek(4)
    ei_class = f.read(1)
    f.seek(5)
    ei_data = f.read(1)
    f.seek(6)
    ei_version = f.read(1)
    f.seek(7)
    ei_osabi = f.read(1)
    f.seek(8)
    ei_abiversion = f.read(1)
    f.seek(16)
    e_type = f.read(2)
    f.seek(18)
    e_machine = f.read(2)
    f.seek(20)
    e_version = f.read(4)
    f.seek(24)
    e_entry = f.read(4)
    f.seek(28)
    e_phoff = f.read(4)
    f.seek(32)
    e_shoff = f.read(4)
    f.seek(36)
    e_flags = f.read(4)
    f.seek(40)
    e_ehsize = f.read(2)
    f.seek(42)
    e_phentsize = f.read(2)
    f.seek(44)
    e_phnum = f.read(2)
    f.seek(46)
    e_shentsize = f.read(2)
    f.seek(48)
    e_shnum = f.read(2)
    f.seek(50)
    e_shstrndx = f.read(2)

    ei_class = int.from_bytes(ei_class, byteorder='little')
    ei_data = int.from_bytes(ei_data, byteorder='little')
    ei_version = int.from_bytes(ei_version, byteorder='little')
    ei_osabi = int.from_bytes(ei_osabi, byteorder='little')
    ei_abiversion = int.from_bytes(ei_abiversion, byteorder='little')
    e_type = int.from_bytes(e_type, byteorder='little')
    e_machine = int.from_bytes(e_machine, byteorder='little')
    e_version = int.from_bytes(e_version, byteorder='little')
    e_entry = int.from_bytes(e_entry, byteorder='little')
    e_phoff = int.from_bytes(e_phoff, byteorder='little')
    e_shoff = int.from_bytes(e_shoff, byteorder='little')
    e_flags = int.from_bytes(e_flags, byteorder='little')
    e_ehsize = int.from_bytes(e_ehsize, byteorder='little')
    e_phentsize = int.from_bytes(e_phentsize, byteorder='little')
    e_phnum = int.from_bytes(e_phnum, byteorder='little')
    e_shentsize = int.from_bytes(e_shentsize, byteorder='little')
    e_shnum = int.from_bytes(e_shnum, byteorder='little')
    e_shstrndx = int.from_bytes(e_shstrndx, byteorder='little')

    return {
        'magic': magic,
        'ei_class': ei_class,
        'ei_data': ei_data,
        'ei_version': ei_version,
        'ei_osabi': ei_osabi,
        'ei_abiversion': ei_abiversion,
        'e_type': e_type,
        'e_machine': e_machine,
        'e_version': e_version,
        'e_entry': e_entry,
        'e_phoff': e_phoff,
        'e_shoff': e_shoff,
        'e_flags': e_flags,
        'e_ehsize': e_ehsize,
        'e_phentsize': e_phentsize,
        'e_phnum': e_phnum,
        'e_shentsize': e_shentsize,
        'e_shnum': e_shnum,
        'e_shstrndx': e_shstrndx
    }

# Only care about i386, x86_64, ARM, and ARM64
elf_machine_types = {
    0x03: 'i386',
    0x3e: 'x86_64',
    0x28: 'ARM',
    0xB7: 'ARM64'
}

e_type_names = {
    0x00: 'ET_NONE',
    0x01: 'ET_REL',
    0x02: 'ET_EXEC',
    0x03: 'ET_DYN',
    0x04: 'ET_CORE',
    0xfe00: 'ET_LOPROC',
    0xffff: 'ET_HIPROC'
}


def print_elf_header(header):
    print('ELF Header:')
    print('  Magic: ' + header['magic'].hex())
    print('  Class: ' + 'ELF64' if header['ei_class'] == 2 else 'ELF32')
    print('  Data: ' + 'LE' if header['ei_data'] == 1 else 'BE')
    print('  Version: ' + str(header['ei_version']))
    print('  OS/ABI: ' + str(header['ei_osabi']))
    print('  ABI Version: ' + str(header['ei_abiversion']))
    print('  Type: ' + e_type_names[header['e_type']])
    print('  Machine: ' + elf_machine_types[header['e_machine']])
    print('  Version: ' + str(header['e_version']))
    print('  Entry point: ' + hex(header['e_entry']))
    print('  Program header offset: ' + hex(header['e_phoff']))
    print('  Section header offset: ' + hex(header['e_shoff']))
    print('  Flags: ' + hex(header['e_flags']))
    print('  Size of this header: ' + hex(header['e_ehsize']))
    print('  Size of program header: ' + hex(header['e_phentsize']))
    print('  Number of program headers: ' + hex(header['e_phnum']))
    print('  Size of section header: ' + hex(header['e_shentsize']))
    print('  Number of section headers: ' + hex(header['e_shnum']))
    print('  Section header string table index: ' + hex(header['e_shstrndx']))


elf_file = open(sys.argv[1], 'rb')
header = parse_elf_header(elf_file)
print_elf_header(header)