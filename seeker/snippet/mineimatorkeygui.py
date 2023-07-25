#date: 2023-07-25T17:05:01Z
#url: https://api.github.com/gists/1aebf68b38314941ccc6188fa47837fd
#owner: https://api.github.com/users/NomadWithoutAHome

import sys
import random
from PyQt5 import QtWidgets, QtGui, QtCore
import pyperclip


class KeyGenerator(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Key Generator')
        self.setStyleSheet('background-color: #303030; color: white;')
        self.setFixedSize(261, 170)  # set fixed window size

        main_layout = QtWidgets.QVBoxLayout()
        self.setLayout(main_layout)

        self.key_label = QtWidgets.QLabel()
        self.key_label.setStyleSheet('font-size: 20px; margin-bottom: 20px;')
        main_layout.addWidget(self.key_label)

        generate_button = QtWidgets.QPushButton('Generate Key')
        generate_button.setStyleSheet('''
            QPushButton {
                background-color: #53C68C;
                color: black;
                font-weight: bold;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 5px;
                margin-right: 10px;
            }
            QPushButton:pressed {
                background-color: #37805A;
            }
        ''')
        generate_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        generate_button.clicked.connect(self.generate_key)
        main_layout.addWidget(generate_button)

        copy_button = QtWidgets.QPushButton('Copy to Clipboard')
        copy_button.setStyleSheet('''
            QPushButton {
                background-color: #53C68C;
                color: black;
                font-weight: bold;
                font-size: 16px;
                padding: 10px 20px;
                border-radius: 5px;
                margin-right: 10px;
            }
            QPushButton:pressed {
                background-color: #37805A;
            }
        ''')
        copy_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        copy_button.clicked.connect(self.copy_key)
        main_layout.addWidget(copy_button)

        with open("style.css", "r") as f:
            self.setStyleSheet(f.read())

    def generate_key(self):
        keystr = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

        while True:
            key = ""

            for _ in range(4):
                pos = random.randint(0, len(keystr)-1)
                key += keystr[pos] + keystr[-pos-1]

            if all(keystr.find(key[i]) == len(keystr) - keystr.find(key[-i-1]) - 1 for i in range(4)):
                self.key_label.setText(f'Your key: {key.upper()}')
                self.generated_key = key.upper()
                break

    def copy_key(self):
        pyperclip.copy(self.generated_key)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    key_generator = KeyGenerator()
    key_generator.show()
    sys.exit(app.exec_())