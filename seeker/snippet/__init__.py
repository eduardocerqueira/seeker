#date: 2024-02-23T16:59:51Z
#url: https://api.github.com/gists/93f072c2a53a814314d7739e0e76e6ad
#owner: https://api.github.com/users/lbianchi-lbl

"""
This file represents where where the numpy types registration in Pyomo should be triggered, e.g. `pyomo.environ` or the appropriate alternative
"""

import sys
from types import ModuleType
from typing import Callable


ModuleHook = Callable[[ModuleType], None]


def _do_numpy_registration(numpy: ModuleType):
    # stand-in for what pyomo needs to do once numpy is imported
    ...


def _install_import_machinery_for_future_numpy_import(after_numpy_imported: ModuleHook):
    from import_machinery import (
        InstallHooksFinder,
        ModuleExecHooks,
        ModuleHooksLoader,
    )

    hooks_for_numpy = ModuleExecHooks(post=after_numpy_imported)
    finder = InstallHooksFinder(
        name_to_hooks={"numpy": hooks_for_numpy},
        factory=ModuleHooksLoader
    )
    finder.enable()


def _set_up_numpy_registration():
    try:
        numpy = sys.modules["numpy"]
    except KeyError:
        # numpy hasn't been imported yet, so we install the import machinery in preparation for a possible future import
        _install_import_machinery_for_future_numpy_import(_do_numpy_registration)
    else:
        # numpy has already been imported when this check is run, so just do the registration now
        _do_numpy_registration(numpy)


_set_up_numpy_registration()