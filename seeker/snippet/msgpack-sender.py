#date: 2022-09-16T23:03:25Z
#url: https://api.github.com/gists/e76f067aad0ac425c9f9008db94e143c
#owner: https://api.github.com/users/SapiensAnatis

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import msgpack
import json
import urllib.parse
import requests

DEFAULT_BASE_URL = "https://localhost:5001/"
REQUEST_HEADERS = {
  'Content-Type': 'application/octet-stream'
}
CERT_PATH = False

class URLInput(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.httpMethod = QComboBox()
        self.httpMethod.addItems(["POST", "GET"])
        layout.addWidget(self.httpMethod, 1)

        self.baseURLInput = QLineEdit(DEFAULT_BASE_URL)
        layout.addWidget(self.baseURLInput, 4)

        self.endpointInput = QComboBox()
        self.endpointInput.addItem("tool/get_service_status")
        self.endpointInput.setCurrentIndex(0)
        self.endpointInput.setEditable(True)
        layout.addWidget(self.endpointInput, 8)

        layout.setContentsMargins(0, 0, 0, 0)
    
    def get_url(self):
        return urllib.parse.urljoin(self.baseURLInput.text(), self.endpointInput.currentText())
    
    def add_endpoint(self, endpoint: str):
        self.endpointInput.addItem(endpoint)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("msgpack sender")
        self.setFixedSize(QSize(500, 700))

        layout = QVBoxLayout()
        self.PopulateWidgets(layout)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)
    
    def PopulateWidgets(self, layout):
        # Initialize widgets as class properties before adding to layout if they will need to be referenced later
        layout.addWidget(QLabel("URL:"))

        self.urlInput = URLInput()
        layout.addWidget(self.urlInput)

        layout.addWidget(QLabel("JSON request body:"))

        self.requestText = QPlainTextEdit("{}")
        layout.addWidget(self.requestText)

        self.sendButton = QPushButton("Send request")
        self.sendButton.clicked.connect(self.SendRequest)
        layout.addWidget(self.sendButton)

        self.responseStatusLabel = QLabel("Response status code:")
        layout.addWidget(self.responseStatusLabel)

        layout.addWidget(QLabel("Response body:"))

        self.responseText = QPlainTextEdit()
        layout.addWidget(self.responseText)

    def SendRequest(self):
        json_dict = {}
        try:
            json_dict = json.loads(self.requestText.toPlainText())
        except Exception as e:
            error_dialog = QErrorMessage()
            error_dialog.showMessage(f'Could not deserialize request body: {e}')
            error_dialog.exec_()
            return
        
        payload = msgpack.packb(json_dict)
        print('payload', payload)
        url = self.urlInput.get_url()
        print('url', url)

        response = None
        try:
            response = requests.request(
                self.urlInput.httpMethod.currentText(),
                url, 
                headers=REQUEST_HEADERS, 
                data=payload, 
                verify=CERT_PATH)
        except Exception as e:
            error_dialog = QErrorMessage()
            error_dialog.showMessage(f'Error while sending request: {e}')
            error_dialog.exec_()
            return
        
        self.responseStatusLabel.setText(f"Response status code: {response.status_code} ({response.reason})")
        
        if response.ok:
            response_dict = msgpack.unpackb(response.content)
            self.responseText.setPlainText(json.dumps(response_dict, indent=4))
        else:
            self.responseText.setPlainText("")

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()