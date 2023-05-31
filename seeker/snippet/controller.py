#date: 2023-05-31T17:01:56Z
#url: https://api.github.com/gists/bd31b19ceacd11c64c6acae0a457d8d9
#owner: https://api.github.com/users/JKearnsl

from PyQt6.QtGui import QMouseEvent

from src.views.main import MainView


class MainController:

    def __init__(self, model: 'MainModel'):
        self.model = model
        self.view = MainView(self, self.model)

        self.view.show()

        # states
        self.old_pos = None

    def add_tab(self):
        print('add_tab')

    def show_maximized(self):
        if self.view.isMaximized():
            self.view.showNormal()
        else:
            self.view.showMaximized()

    def header_mouse_pressed(self, event: QMouseEvent):
        self.old_pos = event.globalPosition()

    def header_mouse_moved(self, event: QMouseEvent):
        delta = (event.globalPosition() - self.old_pos).toPoint()
        print("-------------------")
        print(self.view.x(), self.view.y())
        self.view.move(self.view.x() + delta.x(), self.view.y() + delta.y())
        self.old_pos = event.globalPosition()
        self.view.update()
        print(self.view.x(), self.view.y())