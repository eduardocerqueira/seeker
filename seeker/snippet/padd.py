#date: 2021-09-21T17:05:21Z
#url: https://api.github.com/gists/6f682ba459620b58d306a548619789db
#owner: https://api.github.com/users/dpoulopoulos

import ctypes
import pathlib

if __name__ == "__main__":
    # load the lib
    libname = pathlib.Path().absolute() / "libcadd.so"
    c_lib = ctypes.CDLL(libname)
    
    x, y = 6, 2.3

    # define the return type
    c_lib.cadd.restype = ctypes.c_float
    # call the function with the correct argument types
    res = c_lib.cadd(x, ctypes.c_float(y))
    print(f"In Python: int: {x} float {y:.1f} return val {res:.1f}")