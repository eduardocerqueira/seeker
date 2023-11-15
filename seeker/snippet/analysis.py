#date: 2023-11-15T17:02:39Z
#url: https://api.github.com/gists/73f245ec786b52005006a29c710c12ad
#owner: https://api.github.com/users/ipcoder

from __future__ import annotations

from typing import Union, Callable, Type, Sequence

import pandas as pd
import wrapt

from inu.pipelines.benchmark import Benchmark
from inu.ds.label import Keys, Labels
from holoviews.core import ViewableElement, Params
import panel as pn


pn.pane.Matplotlib.object

class Selector:
    selected: Labels

    def select(self, val): ...


class Analyzer:
    selectors: dict[str, Selector]
    _visualizers: dict[str, ViewableElement]

    def __init_subclass__(cls, **kwargs):
        cls._visualizers = {}


def create_viewable(fnc: Callable, params: Params) -> ViewableElement:
    ...


def visualizer(fnc=None, name: str = None, *, params: dict = None):
    @wrapt.decorator
    def wrapper(wrapped: Callable, instance: Type[Analyzer], args, kwargs):
        instance._visualizers[name or wrapped.__name__] = create_viewable(wrapped, params)
    return wrapper


# ----------------------------------------------
from .adapters.disparity import visual as viz


class TableSelector(Selector):
    def table(self):
        pass


class BundleSelector(Selector):

    def __init__(self, df: pd.DataFrame, bundle: Sequence):
        self.keys = Keys(bundle)
        self.gb = df.groupby(self.keys)
        self.selected = self.select(next(iter(self.gb.bundles)))

    def select(self, labels: tuple) -> Labels:
        self.selected = self.keys(labels)
        return self.selected.copy()

    def table(self):
        return self.gb


class DepthBenchAnalyzer(Analyzer):

    def __int__(self, bmk: Benchmark):
        self.benchmark = bmk
        self.images = bmk.evaluated_ds()

        self.selectors = dict(
            bundle=BundleSelector(self.images.db, self.images.bundle)
        )

    @visualizer('images')
    def show_images(self):
        ims = self.images.qix(**self.selectors['bundle'].selected)
        return viz.show_images(ims)


class AnlyzerGUI:

    def __init__(self, a: Analyzer, layout):
        self.layout = layout
        self._show_visualizers(a.visualizers())

        for name, sel in a.selectors:
            self._show_selector(sel)

    def _show_visualizers(self, vs):
        """
        Create list control with single select over visualizers
        on_select

        :param vs:
        :return:
        """
        pass

    def _show_selector(self, sel: Selector):
        """
        Recognize Selector Type and create its Control accordingly.
        """

    def _change_visualizer(self, name: str):
        """
        Redraw
        :param name:
        :return:
        """


