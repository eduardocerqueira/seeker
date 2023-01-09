#date: 2023-01-09T17:12:50Z
#url: https://api.github.com/gists/340319da23bde512024801aa11a1a111
#owner: https://api.github.com/users/afizs

import sys
from PyQt5.QtWidgets import QApplication, QCalendarWidget, QMainWindow

class CalendarWidget(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create a calendar widget and set it as the central widget
        calendar = QCalendarWidget(self)
        self.setCentralWidget(calendar)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    calendar = CalendarWidget()
    calendar.show()
    sys.exit(app.exec_())
