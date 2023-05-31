#date: 2023-05-31T17:01:56Z
#url: https://api.github.com/gists/bd31b19ceacd11c64c6acae0a457d8d9
#owner: https://api.github.com/users/JKearnsl

from PyQt6 import QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QRegion, QPainter, QColor, QGuiApplication
from PyQt6.QtWidgets import QMainWindow

from src.utils.observer import DObserver
from src.utils.ts_meta import TSMeta
from src.views.main.MainWindow import Ui_MainWindow
from src.models.main import MainModel
from src.widgets.tabwidget import TabContainer


class MainView(QMainWindow, DObserver, metaclass=TSMeta):

    def __init__(self, controller, model: MainModel, parent=None):
        super(QMainWindow, self).__init__(parent)
        self.controller = controller
        self.model = model

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        tab_widget = TabContainer()
        self.ui.centralwidget.layout().addWidget(tab_widget)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # Регистрация представлений
        self.model.add_observer(self)

        # События
        self.ui.closeButton.clicked.connect(self.close)
        self.ui.close.triggered.connect(self.close)
        self.ui.minimizeButton.clicked.connect(self.showMinimized)
        self.ui.maximizeButton.clicked.connect(self.controller.show_maximized)
        self.ui.headerWidget.mousePressEvent = self.controller.header_mouse_pressed
        self.ui.headerWidget.mouseMoveEvent = self.controller.header_mouse_moved

    def model_changed(self):
        pass