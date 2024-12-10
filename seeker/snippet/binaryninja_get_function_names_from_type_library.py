#date: 2024-12-10T17:12:47Z
#url: https://api.github.com/gists/04c00a12d64e25a0b51578efa4497751
#owner: https://api.github.com/users/xorhex

def get_lib_functions(dllname: str):
  for lib in bv.platform.type_libraries:
    if lib.name == dllname:
      for f in lib.named_objects:
        yield(f)