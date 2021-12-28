#date: 2021-12-28T17:20:17Z
#url: https://api.github.com/gists/ec4db3aff4981444b37a8e6ca7215934
#owner: https://api.github.com/users/leo60228

from ctypes import windll, c_int, byref
from ctypes.wintypes import FILETIME
import msvcrt
import os

GENERIC_READ = 0x80000000

FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002

OPEN_ALWAYS = 4

FILE_ATTRIBUTE_NORMAL = 0x80

handle = windll.kernel32.CreateFileA(
        b"test.txt",
        c_int(GENERIC_READ),
        c_int(FILE_SHARE_READ | FILE_SHARE_WRITE),
        None,
        c_int(OPEN_ALWAYS),
        c_int(FILE_ATTRIBUTE_NORMAL),
        None)

leave_unchanged = FILETIME(0xFFFFFFFF, 0xFFFFFFFF)
windll.kernel32.SetFileTime(handle, None, byref(leave_unchanged), None)

fd = msvcrt.open_osfhandle(handle, 0)

with os.fdopen(fd) as file:
    print(file.read())
