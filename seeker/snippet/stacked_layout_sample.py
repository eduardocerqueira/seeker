#date: 2024-06-03T16:58:41Z
#url: https://api.github.com/gists/1f85cd2497d67e2839a602f5a15f92c9
#owner: https://api.github.com/users/HundredVisionsGuy

import sys
from PyQt6.QtGui import QFontDatabase, QFont, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Windchill Calculator")
        self.setContentsMargins(12, 12, 12, 12)
        self.resize(320, 240)

        self.layout = QVBoxLayout()

        # create nav buttons
        nav_layout = QHBoxLayout()
        self.next_button = QPushButton(
            QIcon("resources/icons/icons8-next-50.png"),
            "  Next")
        self.next_button.clicked.connect(self.goto_next)
        self.previous_button = QPushButton(
            QIcon("resources/icons/icons8-previous-50.png"),
            " Back")
        self.previous_button.clicked.connect(self.go_back)

        # add nav buttons to the layout
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.previous_button)

        # Create the stacked layout
        self.stacked_layout = QStackedLayout()

        # Create the home screen
        self.home_screen = QWidget()
        self.home_layout = QVBoxLayout()

        # Create the home screen widgets
        self.home_label = QLabel("Home Screen")
        self.home_label.setFont(QFont("Calibri", 20, 1))

        # Add home layout and widgets to stacked layout
        self.home_layout.addWidget(self.home_label)
        self.home_screen.setLayout(self.home_layout)
        self.stacked_layout.addWidget(self.home_screen)

        # Create Page 2
        self.page2_screen = QWidget()
        self.page2_layout = QVBoxLayout()
        self.page2_label = QLabel("Page 2 Screen")
        self.page2_label.setFont(QFont("Calibri", 20, 1))

        # Add page 2 layout and widgets to stacked layout
        self.page2_layout.addWidget(self.page2_label)
        self.page2_screen.setLayout(self.page2_layout)
        self.stacked_layout.addWidget(self.page2_screen)

        # Add nav and stacked layouts to the main layout
        self.layout.addLayout(nav_layout)
        self.layout.addLayout(self.stacked_layout)

        self.setLayout(self.layout)

    def set_font(self, font_name: str) -> None:
        font_dir = "resources/fonts/"
        font_path = font_dir + font_name
        success = QFontDatabase.addApplicationFont(font_path)

        # if it failed to add the font
        if success == -1:
            print(f"{font_name} not loaded.\nTried path `{font_path}`")

    def goto_next(self) -> None:
        self.stacked_layout.setCurrentIndex(
            self.stacked_layout.currentIndex() + 1
            )

    def go_back(self) -> None:
        self.stacked_layout.setCurrentIndex(
            self.stacked_layout.currentIndex() - 1
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyle("Fusion")
    window = MainWindow()
    window.show()

    app.exec()
