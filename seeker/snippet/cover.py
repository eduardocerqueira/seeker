#date: 2023-04-05T17:06:55Z
#url: https://api.github.com/gists/c5bb2c3fc0ae9054949b10a1612f5169
#owner: https://api.github.com/users/Tired-Fox

import webbrowser
from pathlib import Path

path = Path('htmlcov/index.html').resolve()
if path.is_file():
    webbrowser.open_new_tab(f"file://{path.as_posix()}")
