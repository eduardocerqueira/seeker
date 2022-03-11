#date: 2022-03-11T17:03:05Z
#url: https://api.github.com/gists/87386f6471824dfc3c6d05e1bbfc95b1
#owner: https://api.github.com/users/angelo-soyannwo

import os
import sys
import fcntl
import hashlib
import platform
from cffi import FFI

def suffix():
  if platform.system() == "Darwin":
    return ".dylib"
  else:
    return ".so"

def ffi_wrap(name, c_code, c_header, tmpdir="/tmp/ccache", cflags="", libraries=None):
  if libraries is None:
    libraries = []

  cache = name + "_" + hashlib.sha1(c_code.encode('utf-8')).hexdigest()
  try:
    os.mkdir(tmpdir)
  except OSError:
    pass

  fd = os.open(tmpdir, 0)
  fcntl.flock(fd, fcntl.LOCK_EX)
  try:
    sys.path.append(tmpdir)
    try:
      mod = __import__(cache)
    except Exception:
      print(f"cache miss {cache}")
      compile_code(cache, c_code, c_header, tmpdir, cflags, libraries)
      mod = __import__(cache)
  finally:
    os.close(fd)

  return mod.ffi, mod.lib


def compile_code(name, c_code, c_header, directory, cflags="", libraries=None):
  if libraries is None:
    libraries = []

  ffibuilder = FFI()
  ffibuilder.set_source(name, c_code, source_extension='.cpp', libraries=libraries)
  ffibuilder.cdef(c_header)
  os.environ['OPT'] = "-fwrapv -O2 -DNDEBUG -std=c++1z"
  os.environ['CFLAGS'] = cflags
  ffibuilder.compile(verbose=True, debug=False, tmpdir=directory)


def wrap_compiled(name, directory):
  sys.path.append(directory)
  mod = __import__(name)
  return mod.ffi, mod.lib