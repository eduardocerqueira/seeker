#date: 2025-01-13T16:56:48Z
#url: https://api.github.com/gists/789fc3949e89b965946e84ebae9e2971
#owner: https://api.github.com/users/l1asis

"""
File: ctypes_wrapper.py
Author: Volodymyr Horshenin
"""

import ctypes
import ctypes.wintypes as wintypes

user32 = ctypes.WinDLL("user32")
kernel32 = ctypes.WinDLL("kernel32")


# https://learn.microsoft.com/en-us/windows/win32/dataxchg/standard-clipboard-formats#CF_UNICODETEXT
class StandardClipboardFormats:
    CF_BITMAP = 2
    CF_DIB = 8
    CF_DIBV5 = 17
    CF_DIF = 5
    CF_DSPBITMAP = 0x0082
    CF_DSPENHMETAFILE = 0x008E
    CF_DSPMETAFILEPICT = 0x0083
    CF_DSPTEXT = 0x0081
    CF_ENHMETAFILE = 14
    CF_GDIOBJFIRST = 0x0300
    CF_GDIOBJLAST = 0x03FF
    CF_HDROP = 15
    CF_LOCALE = 16
    CF_METAFILEPICT = 3
    CF_OEMTEXT = 7
    CF_OWNERDISPLAY = 0x0080
    CF_PALETTE = 9
    CF_PENDATA = 10
    CF_PRIVATEFIRST = 0x0200
    CF_PRIVATELAST = 0x02FF
    CF_RIFF = 11
    CF_SYLK = 4
    CF_TEXT = 1
    CF_TIFF = 6
    CF_UNICODETEXT = 13
    CF_WAVE = 12


# https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-globalalloc#GMEM_MOVEABLE
class MemoryAllocationAttributes:
    GHND = 0x0042
    GMEM_FIXED = 0x0000
    GMEM_MOVEABLE = 0x0002
    GMEM_ZEROINIT = 0x0040
    GPTR = 0x0040


# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-openclipboard
user32.OpenClipboard.argtypes = (wintypes.HWND,)
user32.OpenClipboard.restype = wintypes.BOOL

# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-closeclipboard
user32.CloseClipboard.argtypes = None
user32.CloseClipboard.restype = wintypes.BOOL

# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setclipboarddata
user32.SetClipboardData.argtypes = (wintypes.UINT, wintypes.HANDLE)
user32.SetClipboardData.restype = wintypes.HANDLE

# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getclipboarddata
user32.GetClipboardData.argtypes = (wintypes.UINT,)
user32.GetClipboardData.restype = wintypes.HANDLE

# https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-emptyclipboard
user32.EmptyClipboard.argtypes = None
user32.EmptyClipboard.restype = wintypes.BOOL

# https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-globalalloc
kernel32.GlobalAlloc.argtypes = (wintypes.UINT, wintypes.DWORD)
kernel32.GlobalAlloc.restype = wintypes.HANDLE

# https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-globalfree
kernel32.GlobalFree.argtypes = (wintypes.HGLOBAL,)
kernel32.GlobalFree.restype = wintypes.BOOL

# https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-globallock
kernel32.GlobalLock.argtypes = (wintypes.HGLOBAL,)
kernel32.GlobalLock.restype = wintypes.LPVOID

# https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-globalunlock
kernel32.GlobalUnlock.argtypes = (wintypes.HGLOBAL,)
kernel32.GlobalUnlock.restype = wintypes.BOOL

# https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-globalsize
kernel32.GlobalSize.argtypes = (wintypes.HGLOBAL,)
kernel32.GlobalSize.restype = ctypes.c_size_t


MEM_COMMIT = 0x00001000  # https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc#MEM_COMMIT
MEM_RESERVE = 0x00002000  # https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc#MEM_RESERVE
PAGE_EXECUTE_READWRITE = 0x40  # https://learn.microsoft.com/en-us/windows/win32/memory/memory-protection-constants#PAGE_EXECUTE_READWRITE

# https://learn.microsoft.com/en-us/windows/win32/api/memoryapi/nf-memoryapi-virtualalloc
kernel32.VirtualAlloc.argtypes = (
    ctypes.c_void_p,  # LPVOID
    ctypes.c_size_t,  # SIZE_T
    ctypes.c_long,  # DWORD
    ctypes.c_long,  # DWORD
)
kernel32.VirtualAlloc.restype = ctypes.c_void_p  # LPVOID

# https://learn.microsoft.com/en-us/windows/win32/devnotes/rtlmovememory
kernel32.RtlMoveMemory.argtypes = (
    ctypes.c_void_p,  # VOID*
    ctypes.c_void_p,  # VOID*
    ctypes.c_size_t,  # SIZE_T
)
kernel32.RtlMoveMemory.restype = None


asm_int_add = (
    b"\x8b\xc1"  # mov eax, ecx
    b"\x83\xc0\x01"  # add eax, 1
    b"\xc3"  # ret
)

asm_int_mul = (
    b"\x8b\xc1"  # mov eax, ecx
    b"\x69\xc0\x02\x00\x00\x00"  # mul, eax, 2
    b"\xc3"  # ret
)

asm_float_add = (
    b"\xF3\x0F\x58\xC1"  # addss xmm0, xmm1
    b"\xC3"  # ret
)


def asm_function(data: bytes, restype, *argtypes):
    memory_buffer = kernel32.VirtualAlloc(
        0,  # lpAddress - NULL
        len(data),  # dwSize
        MEM_COMMIT | MEM_RESERVE,  # flAllocationType
        PAGE_EXECUTE_READWRITE,  # flProtect
    )

    if not memory_buffer:  # VirtualAlloc returned NULL
        print("VirtualAlloc call failed. Error code:", ctypes.GetLastError())
        exit(-1)

    c_buffer = ctypes.c_char_p(data)

    kernel32.RtlMoveMemory(
        memory_buffer, c_buffer, len(data)  # Destination  # Source  # Length
    )

    func = ctypes.cast(
        memory_buffer,
        ctypes.CFUNCTYPE(restype, *argtypes),  # return type, argument type, ...
    )
    return func


def get_clipboard(
    clipboard_format: int = StandardClipboardFormats.CF_UNICODETEXT,
) -> str:
    if user32.OpenClipboard(None):
        handle = user32.GetClipboardData(
            clipboard_format
        )  # handle to a clipboard object
        pointer = kernel32.GlobalLock(handle)
        size = kernel32.GlobalSize(pointer)
        c_buffer = ctypes.create_string_buffer(size)
        ctypes.memmove(c_buffer, pointer, size)
        kernel32.GlobalUnlock(handle)
        return c_buffer.raw.decode("utf-16le")
    else:
        print("OpenClipboard call failed. Error code:", ctypes.GetLastError())
        exit(-1)


def set_clipboard(
    data: str | bytes, clipboard_format: int = StandardClipboardFormats.CF_UNICODETEXT
) -> None:
    if not isinstance(data, bytes):
        data = str(data).encode("utf-16le") + b"\x00"
    if user32.OpenClipboard(None):
        user32.EmptyClipboard()
        handle = kernel32.GlobalAlloc(
            MemoryAllocationAttributes.GMEM_MOVEABLE, len(data) + 1
        )  # handle to a memory object
        pointer = kernel32.GlobalLock(handle)
        ctypes.memmove(pointer, data, len(data))
        kernel32.GlobalUnlock(handle)
        user32.SetClipboardData(clipboard_format, handle)
        user32.CloseClipboard()
        if kernel32.GlobalFree(handle):
            print("GlobalFree call failed. Error code:", ctypes.GetLastError())
            exit(-1)
    else:
        print("OpenClipboard call failed. Error code:", ctypes.GetLastError())
        exit(-1)


if __name__ == "__main__":
    func = asm_function(asm_float_add, ctypes.c_float, ctypes.c_float, ctypes.c_float)
    print(f"{func(13, 13) = :.50f}")
    set_clipboard("Windows Clipboard.")
    print(get_clipboard())
