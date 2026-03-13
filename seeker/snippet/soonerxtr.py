#date: 2026-03-13T17:19:17Z
#url: https://api.github.com/gists/05e643c7034c1a14840d8ffa5e3ae9e1
#owner: https://api.github.com/users/Dobby233Liu

# SoonerXTR: Extracts a HTC EXCA300 system.img
# which presumably uses a modified or early version of YAFFS1
# Author: AyeTSG
# Cleanup by another guy

import os
from ctypes import LittleEndianStructure, c_char, c_uint16, c_uint32
from enum import IntEnum, auto

PAGE_SIZE = 0x800
PAGE_DATA_SIZE = 0x200


def roundr(n, step):
    return ((n - 1) // step + 1) * step


# https://github.com/kempniu/yaffs2/blob/master/yaffs_guts.h


class yaffs_obj_type(IntEnum):
    YAFFS_OBJECT_TYPE_UNKNOWN = 0
    YAFFS_OBJECT_TYPE_FILE = auto()
    YAFFS_OBJECT_TYPE_SYMLINK = auto()
    YAFFS_OBJECT_TYPE_DIRECTORY = auto()
    YAFFS_OBJECT_TYPE_HARDLINK = auto()
    YAFFS_OBJECT_TYPE_SPECIAL = auto()


YAFFS_OBJECTID_ROOT = 1

YAFFS_MAX_NAME_LENGTH = 255
YAFFS_MAX_ALIAS_LENGTH = 159


class yaffs_obj_hdr(LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("type", c_uint32),
        # for everything
        ("parent_obj_id", c_uint32),
        ("sum_no_longer_used", c_uint16),
        ("name", c_char * (YAFFS_MAX_NAME_LENGTH + 1)),
        # for all except hardlinks
        ("yst_mode", c_uint32),
        ("yst_uid", c_uint32),
        ("yst_gid", c_uint32),
        ("yst_atime", c_uint32),
        ("yst_mtime", c_uint32),
        ("yst_ctime", c_uint32),
        ("unknown", c_uint16),  # FORMAT DIFFERENCE
        # file only
        ("file_size_low", c_uint32),
        # hardlink only
        ("equiv_id", c_uint32),
        # symlink only
        ("alias", c_char * (YAFFS_MAX_ALIAS_LENGTH + 1)),
        #
        ("yst_rdev", c_uint32),
    ]


YAFFS_NOBJECT_BUCKETS = 0x100

# these need to be done manually at the moment, but can easily be done
# using the data that gets output to console
# TODO: I think this can be solved by making the reading process two-passed
DIR_NODE_NAMES = {
    (YAFFS_OBJECTID_ROOT - YAFFS_NOBJECT_BUCKETS): "/",
    1: "/bin/",
    49: "/usr/",
    50: "/usr/share/",
    51: "/usr/share/bsk/",
    54: "/usr/share/zoneinfo/",
    57: "/usr/keylayout/",
    62: "/usr/cert/",
    64: "/usr/keychars/",
    69: "/fonts/",
    78: "/media/",
    79: "/media/audio/",
    80: "/media/audio/ringtones/",
    89: "/sounds/",
    97: "/lib/",
    98: "/lib/modules/",
    152: "/app/",
    197: "/javalib/",
}


cur_obj_id = YAFFS_NOBJECT_BUCKETS


def read_entry(img):
    global cur_obj_id

    # FORMAT DIFFERENCE: I don't think each page contains spare data
    chunk_data = img.read(PAGE_SIZE)
    if len(chunk_data) == 0:
        raise StopIteration
    oh = yaffs_obj_hdr.from_buffer_copy(chunk_data[:PAGE_DATA_SIZE])

    # if it's a file...
    if oh.type == yaffs_obj_type.YAFFS_OBJECT_TYPE_FILE:
        # -- GET DIR IDENTIFIER
        dir_ident = oh.parent_obj_id
        print(f"PARENT OBJ ID: {dir_ident}")

        # -- GET FILE NAME
        file_name = oh.name.rstrip(b"\x00").decode("ascii")
        print("FILE: " + file_name)

        # -- GET FILE SIZE
        file_size_low = oh.file_size_low

        # -- READ THE BINARY!
        path = "mtd5_system" + DIR_NODE_NAMES[dir_ident - YAFFS_NOBJECT_BUCKETS] + file_name
        print(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            f.write(img.read(file_size_low))

    # if it's a dir...
    elif oh.type == yaffs_obj_type.YAFFS_OBJECT_TYPE_DIRECTORY:
        # -- GET DIR IDENTIFIER
        parent_obj_id = oh.parent_obj_id
        print(f"PARENT OBJ ID: {parent_obj_id}")

        # -- GET DIR NAME
        dir_name = oh.name.rstrip(b"\x00").decode("ascii")
        print("DIR: " + dir_name)

    else:
        return

    ## == GET NEXT ENTRY
    img.seek(roundr(img.tell(), PAGE_SIZE))

    print("OBJ ID: " + str(cur_obj_id))
    cur_obj_id += 1
    print()


with open("mtd5_system.img", "rb") as img:
    while True:
        read_entry(img)
