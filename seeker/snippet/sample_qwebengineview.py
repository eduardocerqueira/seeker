#date: 2022-03-09T17:09:06Z
#url: https://api.github.com/gists/daad575aaa1221378dea8ac16dafab2e
#owner: https://api.github.com/users/yekyam

import sys
from PySide6 import QtWidgets
from PySide6.QtWebEngineWidgets import QWebEngineView

if __name__ == "__main__":
	app = QtWidgets.QApplication([])
	widget = QWebEngineView()
	widget.resize(600, 600)
  
	with open("test.html") as f:
		widget.setHtml(''.join([l for l in f]))
    
	widget.show()

	sys.exit(app.exec())