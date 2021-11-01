#date: 2021-11-01T16:55:12Z
#url: https://api.github.com/gists/2347edd34a44c6738f48a339fbab496e
#owner: https://api.github.com/users/WWWCourses

import sys
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg


class MainWindow(qtw.QWidget):

	def __init__(self , *args, **kwargs):
		super().__init__(*args, **kwargs)

		self.setup_UI()

		self.show();

	def setup_UI(self):
		self.setWindowTitle('Signals And Slots')
		self.setGeometry(500, 300, 600, 300)

		self.leUserName = qtw.QLineEdit(self)
		self.lePassword = qtw.QLineEdit(self)

		# create Form Group Box
		# form_groupbox = self.create_form_groupbox()
		self.create_form_groupbox()

		# create Buttons Layout
		self.create_buttons_layout()

		# create main layout
		main_layout = qtw.QVBoxLayout(self)
		main_layout.addWidget(self.form_groupbox)
		main_layout.addLayout(self.buttons_layout)

	def create_form_groupbox(self):
		self.form_groupbox = qtw.QGroupBox('Login Form')

		form_layout = qtw.QFormLayout(self)
		form_layout.addRow('Name:',self.leUserName )
		form_layout.addRow('Password:', self.lePassword)
		self.form_groupbox.setLayout(form_layout)

	def create_buttons_layout(self):
		self.buttons_layout = qtw.QHBoxLayout()
		btn_OK = qtw.QPushButton('OK')
		btn_Cansel = qtw.QPushButton('Cancel')
		self.buttons_layout.addWidget(btn_OK)
		self.buttons_layout.addWidget(btn_Cansel)

	def on_close(self):
		# connect btn's click signal with close
		pass






if __name__ == '__main__':
	app = qtw.QApplication(sys.argv);

	window = MainWindow()

	sys.exit(app.exec_())
