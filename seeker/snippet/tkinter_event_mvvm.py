#date: 2023-02-16T16:43:54Z
#url: https://api.github.com/gists/d2b204537eb099fd08778cd19c9f7844
#owner: https://api.github.com/users/stepanzh

# Basic event handling approach to keep Model-View-ViewModel architecture.

import tkinter as tk
import tkinter.ttk as ttk
from typing import Callable, List
from random import randint


Action = Callable[[], None]

# There is no separation between event and its handler,
# but this class is enough to attach Actions and trigger them.
class EventHandler:
    def __init__(self):
        self._event_var = tk.BooleanVar()

    def attach(self, action: Action):
        self._event_var.trace('w', lambda x, y, z: action())

    def trigger(self):
        self._event_var.set(True)


# ViewModel has public property, which can be changed from somewhere.
# Whenever ingredients are changed, viewmodel triggers its handler.
class FooViewModel:
    def __init__(self):
        self._ingredients: List[str] = []
        self.update_handler = EventHandler()

    @property
    def ingredients(self):
        return self._ingredients

    @ingredients.setter
    def ingredients(self, value):
        self._ingredients = value
        self.update_handler.trigger()


# The view attaches handler for update itself.
# So, the viewmodel and view are bounded through EventHandler.
class FooView(ttk.Frame):
    def __init__(self, parent, viewmodel: FooViewModel):
        super().__init__(parent)
        self.ingredients = ttk.Label(self)
        self.ingredients.pack()

        viewmodel.update_handler.attach(lambda: self._update_ingredients(viewmodel.ingredients))

    def _update_ingredients(self, ingredients: List[str]):
        string = '\n'.join(ingredients)
        self.ingredients.configure(text=string)


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.viewmodel = FooViewModel()

        view = FooView(self, self.viewmodel)
        # For demo purposes button is placed here, but it should be in view.
        button = ttk.Button(self, text='Click me', command=self.set_some_ingredients_command)

        view.pack(fill=tk.BOTH, expand=1)
        button.pack(fill=tk.BOTH, expand=1)

    def set_some_ingredients_command(self):
        # Finally!
        self.viewmodel.ingredients = [
            f'Banana {randint(0, 10)}',
            f'Milk {randint(0, 10)}',
            f'Sugar {randint(0, 10)}',
            f'Icecream {randint(0, 10)}',
        ]


# Also, we can define a tk.StringVar in the viewmodel and trace it in the view.
# But, using EventHandler, there is clean separation between an object (ingredients list) and its represantion in ui.

if __name__ == "__main__":
    app = App()
    app.mainloop()