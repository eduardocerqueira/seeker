#date: 2022-06-01T17:02:34Z
#url: https://api.github.com/gists/8fefcfa4d9e374357a662bb78e547a5b
#owner: https://api.github.com/users/hisashi-ito

from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = Extension("loop", sources=["loop.pyx"], extra_compile_args=["-O3"])
setup(name="loop", ext_modules=cythonize([ext]))
