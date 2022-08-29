#date: 2022-08-29T17:18:09Z
#url: https://api.github.com/gists/ba59d7409091a1386fdc310943b60f38
#owner: https://api.github.com/users/dot1mav

from flask import Flask
from flask_sqlalchemy.model import DefaultMeta as Model
from typing import Dict, Optional

class ProxyModel:
    app: Flask
    module_name: str
    models: Dict[str, Model]

    def __init__(self, app: Optional[Flask] = None, module_name: str = "models") -> None:
        self.module_name = module_name

        self.models = {}

        if app:
            self.init_app(app)

    def __call__(self, model_name: str, blueprint_name: Optional[str] = None) -> Model:
        if not isinstance(model_name, str):
            raise TypeError()

        if blueprint_name:
            key = f"{blueprint_name.lower()}_{model_name.lower()}"
        else:
            key = model_name.lower()

        if key not in self.models:
            raise ValueError()

        return self.models[key]


    def __getitem__(self, model_name: str) -> Model:
        return self.__call__(model_name)

    def init_app(self, app: Flask) -> None:
        self.app = app

        self._load_from_app()
        self._load_from_blueprints()

    @staticmethod
    def _find_models(modules) -> list:
        models = []
        for key in modules.__dict__:
            if isinstance(modules.__dict__[key], Model):
                models.append(modules.__dict__[key])

        return models

    def _load_from_app(self) -> None:
        try:
            models_module = import_module(
                f".{self.module_name}", self.app.import_name)

            models = self._find_models(models_module)

            for model in models:
                self.models.update({model.__name__.lower(): model})

        except ModuleNotFoundError as err:
            self.app.logger.warning(f"{err!r}")

    def _load_from_blueprints(self) -> None:
        for bp in self.app.blueprints:
            try:
                bp_models = import_module(
                    f".{self.module_name}", self.app.blueprints[bp].import_name)

                models = self._find_models(bp_models)

                for model in models:
                    self.models.update(
                        {f"{bp_models.__name__.split('.')[-2].lower()}_{model.__name__.lower()}": model})

            except ModuleNotFoundError as err:
                self.app.logger.warning(f"{err!r}")