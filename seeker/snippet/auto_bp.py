#date: 2022-08-29T17:15:13Z
#url: https://api.github.com/gists/03efe7afcbda405fd954553516b2499f
#owner: https://api.github.com/users/dot1mav

import os

from importlib import import_module
from flask import Flask
from typing import Optional

class AutoBlueprint:
    app: Flask
    base_directory: str
    ignore_list: list
    blueprints_dir_name: str
    bp_module_name: str
    bp_modules: list

    def __init__(self, app: Optional[Flask] = None, directory: str = "blueprints", module_name: str = "bp",
                 ignore_list: Optional[list] = None) -> None:
        self.blueprints_dir_name = directory
        self.bp_module_name = module_name

        if ignore_list is None:
            self.ignore_list = []
        else:
            self.ignore_list = ignore_list

        self.ignore_list.append("__pycache__")

        self.bp_modules = []

        if app:
            self.init_app(app)

    def init_app(self, app: Flask, load: bool = True) -> None:
        self.app = app

        self.base_directory = os.path.abspath(os.path.dirname(__file__))

        if load:
            self.load_module()

    def load_module(self) -> None:
        self._load_directory()

        for module in self.bp_modules:
            try:
                bp = getattr(module, self.bp_module_name)
                self.app.register_blueprint(bp)
            except AttributeError as err:
                self.app.logger.warning(f"{err!r}")

    def _load_directory(self) -> None:
        for bp_module in os.listdir(os.path.join(self.base_directory, self.blueprints_dir_name)):
            bp_dir = os.path.join(self.base_directory,
                                  self.blueprints_dir_name, bp_module)
            if bp_module not in self.ignore_list and os.path.isdir(bp_dir):
                try:
                    self.bp_modules.append(import_module(
                        f'.{bp_module}', f'{self.app.import_name}.{self.blueprints_dir_name}'))
                except ModuleNotFoundError as err:
                    self.app.logger.warning(f"{err!r}")