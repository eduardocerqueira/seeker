#date: 2025-05-29T17:11:35Z
#url: https://api.github.com/gists/779be4a2fcd334346e2b26e495882af6
#owner: https://api.github.com/users/LonMcGregor

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QInputDialog, QTextEdit, QListWidget, QLineEdit, QListWidgetItem
from krita import Krita
from PyQt5.QtCore import Qt
import re

mykrita = Krita.instance()
allkritacmds = mykrita.actions()
currentitems = []
currentselection= 0

class CustomLineEdit(QLineEdit):
    def keyPressEvent(self, e):
        global currentitems
        global currentselection
        if len(currentitems) > 0:
            if e.key() == Qt.Key_Up:
                currentitems[currentselection].setSelected(False)
                currentselection = (currentselection - 1) % len(currentitems)
                currentitems[currentselection].setSelected(True)
            if e.key() == Qt.Key_Down:
                currentitems[currentselection].setSelected(False)
                currentselection = (currentselection + 1) % len(currentitems)
                currentitems[currentselection].setSelected(True)
        super(CustomLineEdit, self).keyPressEvent(e)

def match(action, text):
    terms = text.lower().split(' ')
    for term in terms:
        if term not in action.text().lower() or term not in action.toolTip().lower():
            return False
    return True

def onTextChanged(cmdsearch):
    global currentitems
    global currentselection
    boxlist.clear()
    if cmdsearch is None or cmdsearch == '':
        return
    validcommand = [x for x in allkritacmds if match(x, cmdsearch)]
    currentitems = []
    currentselection = 0
    for item in validcommand:
        itemrow = QListWidgetItem()
        itemrow.setData(1, item.objectName())
        stext = item.shortcut().toString()
        itemrow.setToolTip(item.toolTip())
        # need to remove excess ampersands used as menu accelerators
        displaytext = re.sub('&(.)', r'\1', item.text())
        if len(stext) > 0:
            itemrow.setText(displaytext + ' ('+item.shortcut().toString()+')')
        else:
            itemrow.setText(displaytext)
        boxlist.addItem(itemrow)
        currentitems.append(itemrow)
    if len(currentitems) > 0:
        currentitems[0].setSelected(True)
        
def makeSelection():
    selection = boxlist.selectedItems()[0]
    mykrita.action(selection.data(1)).trigger()
    nd.done(0)

boxlayout = QVBoxLayout()
boxtextin = CustomLineEdit()
boxtextin.textChanged.connect(onTextChanged)
boxtextin.returnPressed.connect(makeSelection)
boxlist = QListWidget()
boxlayout.addWidget(boxtextin)
boxlayout.addWidget(boxlist)

nd = QDialog()
nd.setLayout(boxlayout)
nd.setWindowTitle('Quick Commands')
nd.exec_()