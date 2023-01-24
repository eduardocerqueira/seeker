#date: 2023-01-24T17:08:31Z
#url: https://api.github.com/gists/38b25686855b10082cf9e9a03124bfec
#owner: https://api.github.com/users/carlos-villavicencio-adsk

# ------------------------------------------------------------------------------
# installed debugpy with pip install -t to this directory
import sys; sys.path.append("C:/Users/villavc/Code/python-libs")
import debugpy
# got it from pyenv python >>> print(sys.executable)
debugpy.configure(python=r"C:\Users\villavc\.pyenv\pyenv-win\versions\3.9.13\python.exe")
try:
    debugpy.listen(("127.0.0.1", 5678))
except RuntimeError as e:
    if "Only one usage of each socket address" in str(e):
        pass
debugpy.wait_for_client()
debugpy.breakpoint()
# ------------------------------------------------------------------------------
# ...