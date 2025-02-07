#date: 2025-02-07T17:00:52Z
#url: https://api.github.com/gists/e406c198fd4c846bc2eb0d5b78be859c
#owner: https://api.github.com/users/marionlb

import rich

RICH_WIDTH = 140

consoleargs = {"force_jupyter": False, "width": RICH_WIDTH}
rich.reconfigure(**consoleargs)

rich.pretty.install(indent_guides=True, expand_all=True)
rich.traceback.install(
    # show_locals=True,
    locals_hide_sunder=True,
    console=rich.console.Console(**(consoleargs | {"stderr": True})),
    width=RICH_WIDTH,
    code_width=120,
)
