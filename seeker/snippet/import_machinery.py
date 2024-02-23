#date: 2024-02-23T16:59:51Z
#url: https://api.github.com/gists/93f072c2a53a814314d7739e0e76e6ad
#owner: https://api.github.com/users/lbianchi-lbl

"""
The code in this module is completely generic, i.e. Pyomo- and numpy-agnostic
"""

from dataclasses import dataclass
import importlib
import sys
from types import ModuleType  # imported module objects are instances of this
from typing import (
  Callable,
  Mapping,
)


ModuleName = str
ModuleHook = Callable[[ModuleType], None]


def _do_nothing(*args, **kwargs) -> None:
    pass


@dataclass
class ModuleExecHooks:
    pre: ModuleHook = _do_nothing
    post: ModuleHook = _do_nothing


@dataclass
class ModuleHooksLoader(importlib.abc.Loader):
    """
    A wrapper class around a concrete Loader instance ``wrapped``.

    Run specified hook functions before and/or after module execution
    (exec_module() is similar to __init__) and defer to the wrapped instance otherwise.
    """

    wrapped: importlib.abc.Loader
    hooks: ModuleExecHooks

    def module_repr(self, module: ModuleType) -> str:
        # normal behavior through self.wrapped
        return self.wrapped.module_repr(module)

    def create_module(self, spec) -> ModuleType:
        # normal behavior through self.wrapped
        return self.wrapped.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.hooks.pre(module)
        # normal behavior through self.wrapped
        self.wrapped.exec_module(module)
        self.hooks.post(module)


GetLoaderWithHooks = Callable[[importlib.abc.Loader, ModuleExecHooks], importlib.abc.Loader]


@dataclass
class InstallHooksFinder(importlib.abc.MetaPathFinder):
    """
    Custom Finder class to replace loaders with a ModuleHooksLoader instance for the specified modules.

    To activate it, it must be placed before default Finders (i.e. sys.meta_path.insert(0)).
    """

    name_to_hooks: Dict[ModuleName, ModuleExecHooks]
    factory: GetLoaderWithHooks = ModuleHooksLoader

    def find_spec(self, *args, **kwargs):
        spec = importlib.machinery.PathFinder.find_spec(*args, **kwargs)
        if spec is None:
            # returning None will cause the import machinery to proceed with the next Loader in sys.meta_path
            return
        try:
            hooks_for_name: ModuleExecHooks = self.name_to_hooks[spec.name]
        except KeyError:
            # spec doesn't correspond to a registered item, so we don't modify it
            pass
        else:
            new_loader = self.factory(spec.loader, hooks_for_name)
            spec.loader = new_loader
        return spec

    def enable(self):
        sys.meta_path.insert(0, self)

    def disable(self):
        sys.meta_path.remove(self)