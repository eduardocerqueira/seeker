#date: 2022-11-08T17:18:47Z
#url: https://api.github.com/gists/5d5fdc235672b42d49e6907634064dfa
#owner: https://api.github.com/users/leaver2000

import os
from setuptools import Extension, setup
from Cython.Build import cythonize

os.environ["TEST"] = "TRUE"
TEST = bool(os.environ.get("TEST", False))

compiler_directives: dict[str, int | bool] = {"language_level": 3}
define_macros: list[tuple[str, str]] = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")
]

if TEST:
    # inorder to compile the cython code for test coverage
    # we need to include the following compiler directives...
    compiler_directives.update({"linetrace": True, "profile": True})
    # and include the following trace macros
    define_macros.extend([("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")])

ext_modules = cythonize(
    [
        Extension("app._api", ["app/_api.pyx"], define_macros=define_macros),
    ],
    compiler_directives=compiler_directives,
)

setup(
    ext_modules=ext_modules,
)
