#date: 2022-02-02T17:12:39Z
#url: https://api.github.com/gists/7bab1b37b652ab3d82806540d84d43bc
#owner: https://api.github.com/users/ETKNeil

from cffi import FFI
from ctypes import *
import _cffi_backend

ffi = _cffi_backend.FFI()
ctypes_lib = CDLL("./_native__lib.so",mode = RTLD_GLOBAL)
print(ctypes_lib)
handle=ctypes_lib._handle
print(handle)

# handle=ctypes.cast(handle, ctypes.POINTER(ffi.CData))
handle=ctypes.cast(handle, ctypes.c_void_p)
print(handle)
lib = ffi.dlopen(handle)