#date: 2023-05-31T17:01:56Z
#url: https://api.github.com/gists/bd31b19ceacd11c64c6acae0a457d8d9
#owner: https://api.github.com/users/JKearnsl

import sys

from PyQt6 import QtCore
from PyQt6.QtWidgets import QApplication, QStyleFactory

from src.models.main import MainModel
from src.controllers.main import MainController


def main():
    QApplication.setDesktopSettingsAware(False)
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))

    QtCore.QDir.addSearchPath('icons', 'assets/window/icons')

    model = MainModel()
    controller = MainController(model)

    app.exec()


if __name__ == '__main__':
    sys.exit(main())