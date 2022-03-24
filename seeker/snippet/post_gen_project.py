#date: 2022-03-24T17:16:55Z
#url: https://api.github.com/gists/1ea1c9b227d0d115d68b4e2294ada235
#owner: https://api.github.com/users/rwhitt2049

"""
This is a hack to get a single file out of cookiecutter
as it doesn't natively support single file templates
"""

import os
import pathlib


os.chdir("..")
p = pathlib.Path("{{cookiecutter.file_name}}")
f = p.joinpath("{{cookiecutter.file_name}}.py")
f.rename("./{{cookiecutter.file_name}}.py")
p.rmdir()
