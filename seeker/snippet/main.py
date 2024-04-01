#date: 2024-04-01T17:05:58Z
#url: https://api.github.com/gists/e2f4ad26dee478add4b4804c63f66373
#owner: https://api.github.com/users/HorseCheng

from CheckableComboBox import CheckableComboBox
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(400, 600)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = CheckableComboBox(Form)
        self.frame.setObjectName("frame")
        comunes = ["Select All",'Ameglia', 'Arcola', 'Bagnone']
        self.frame.addItems(comunes)
        self.frame.activated.connect(self.handleActivated)

        self.horizontalLayout.addWidget(self.frame)
        self.xx = QtWidgets.QPushButton(Form)
        self.horizontalLayout.addWidget(self.xx)
        self.xx.clicked.connect(self.handleActivated)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))

    def handleActivated(self):
        print(self.frame.currentData())
        
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    widget = QtWidgets.QWidget()

    widget.show()
    test = Ui_Form()
    test.setupUi(widget)
    sys.exit(app.exec_())
