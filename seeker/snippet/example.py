#date: 2023-08-14T16:50:00Z
#url: https://api.github.com/gists/aa82dd3fdde0ddc15016ba5dec46d7c5
#owner: https://api.github.com/users/rfletchr

import time
from PySide2 import QtGui, QtWidgets, QtCore
from concurrent.futures import ThreadPoolExecutor


class SimpleController(QtCore.QObject):
    progressChanged = QtCore.Signal(float)

    def __init__(self, view=None):
        super().__init__()
        self.view = view or SimpleView()
        self.view.buttonClicked.connect(self.onButtonClicked)

        self.model = QtGui.QStandardItemModel()
        self.view.setModel(self.model)
        self.progressChanged.connect(self.view.setProgress)
        self.executor = ThreadPoolExecutor(max_workers=1)

    def doWork(self):
        """
        This method is executed in a background thread.
        """
        items = [
            self.model.item(i) for i in range(self.model.rowCount())
            if self.model.item(i).checkState() == QtCore.Qt.Checked
        ]

        for index, item in enumerate(items):
            time.sleep(1)
            self.progressChanged.emit((index + 1) / len(items) * 100)

    def onButtonClicked(self):
        """
        - When the button is clicked, submit the work function to the executor, and update the UI.
        - When the work is finished, update the UI again using the callback.
        """

        def callback(*_):
            self.view.onWorkFinished()

        self.view.onWorkStarted()
        # Submit the work function to the executor.
        future = self.executor.submit(self.doWork)

        # add a callback to the future, so that the UI is updated when the work is done.
        future.add_done_callback(callback)

    def populate(self):
        for i in range(10):
            item = QtGui.QStandardItem(f"Item {i}")
            item.setCheckable(True)
            self.model.appendRow(item)

    def show(self):
        self.populate()
        self.view.show()


class SimpleView(QtWidgets.QWidget):
    buttonClicked = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.item_list = QtWidgets.QListView()

        self.button = QtWidgets.QPushButton("Click me")
        self.button.clicked.connect(self.buttonClicked)
        self.progress = QtWidgets.QProgressBar()

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.button)
        layout.addWidget(self.item_list)
        layout.addWidget(self.progress)

    def setModel(self, model):
        self.item_list.setModel(model)

    def setProgress(self, value):
        self.progress.setValue(value)

    def onWorkStarted(self):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.progress.setValue(0)
        self.button.setEnabled(False)

    def onWorkFinished(self):
        QtWidgets.QApplication.restoreOverrideCursor()
        self.button.setEnabled(True)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    controller = SimpleController()
    controller.show()
    app.exec_()
