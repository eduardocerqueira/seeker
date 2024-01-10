#date: 2024-01-10T16:57:10Z
#url: https://api.github.com/gists/97ea9a2998445672e19d83187bed3665
#owner: https://api.github.com/users/Decstasy

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2024 - Dennis Ullrich
A simple GUI for reading *.conf files in /etc/wireguard. It toggles your connection using wg-quick up/down and displays status information while enabled.
Root permissions are required for proper functionality.

License: MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QSlider, QTableWidget, QTableWidgetItem, QSizePolicy, QLayout
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

import sys
import subprocess
import os
import re
import threading

class WireGuardGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.running = True

        self.setWindowTitle("WireGuard VPN Manager")
        self.config_dir = "/etc/wireguard"
        self.current_vpn = ""
        self.refresh_interval = 5  # seconds for status refresh
        self.initUI()

    def closeEvent(self, event):
        self.running = False
        super().closeEvent(event)

    def initUI(self):
        layout = QVBoxLayout()
        self.setMinimumSize(460, 240)

        # Status Table Configuration
        self.status_table = QTableWidget()
        self.status_table.setColumnCount(2)
        self.status_table.horizontalHeader().hide()
        self.status_table.verticalHeader().hide()
        self.status_table.setShowGrid(False)
        # Größenpolitik für die Tabelle
        self.status_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.status_table)

        # Toggle Button Configuration
        self.toggle_button = QPushButton("Toggle VPN")
        self.toggle_button.clicked.connect(self.toggle_vpn)
        self.toggle_button.setStyleSheet("background-color: grey")
        layout.addWidget(self.toggle_button)

        # Connection List Configuration
        self.connection_list = QComboBox()
        self.connection_list.addItems(self.get_config_list())
        active_vpn = self.get_active_vpn()
        if active_vpn:
            index = self.connection_list.findText(active_vpn, Qt.MatchFixedString)
            if index >= 0:
                self.connection_list.setCurrentIndex(index)
                self.toggle_button.setStyleSheet("background-color: green")
                self.connection_list.setEnabled(False)
        else:
            self.toggle_button.setStyleSheet("background-color: grey")
        layout.addWidget(self.connection_list)

        self.setLayout(layout)
        self.update_status_loop()

    def get_config_list(self):
        return sorted([f[:-5] for f in os.listdir(self.config_dir) if f.endswith('.conf')])

    def toggle_vpn(self):
        self.current_vpn = self.connection_list.currentText()
        if not self.current_vpn:
            return

        if self.is_vpn_active(self.current_vpn):
            self.vpn_down(self.current_vpn)
            self.toggle_button.setStyleSheet("background-color: grey")
            self.connection_list.setEnabled(True)
        else:
            self.vpn_up(self.current_vpn)
            self.toggle_button.setStyleSheet("background-color: green")
            self.connection_list.setEnabled(False)

    def is_vpn_active(self, config):
        output = subprocess.check_output(["wg", "show"]).decode()
        if output.strip() == "":
            return False
        return config in output

    def vpn_up(self, config):
        subprocess.run(["wg-quick", "up", config])

    def vpn_down(self, config):
        subprocess.run(["wg-quick", "down", config])

    def get_vpn_status(self):
        output = subprocess.check_output(["wg", "show"]).decode()
        pattern = re.compile(r'(interface: .+|endpoint: .+|allowed ips: .+|latest handshake: .+|transfer: .+|persistent keepalive: .+)')
        relevant_lines = pattern.findall(output)
        return relevant_lines

    def get_active_vpn(self):
        output = subprocess.check_output(["wg", "show"]).decode()
        if output.strip() == "":
            return False
        pattern = re.compile(r'interface: (\S+)')
        match = pattern.search(output)
        return match.group(1) if match else False

    def update_status_loop(self):
        if not self.running:
            return
        self.status_table.clear()
        status_lines = self.get_vpn_status()
        self.status_table.setRowCount(len(status_lines))
        for i, line in enumerate(status_lines):
            key, value = line.split(": ")
            self.status_table.setItem(i, 0, QTableWidgetItem(key))
            self.status_table.setItem(i, 1, QTableWidgetItem(value))
        self.status_table.resizeColumnsToContents()
        self.status_table.resizeRowsToContents()
        self.toggle_button.setStyleSheet("background-color: green" if self.is_vpn_active(self.current_vpn) else "background-color: grey")
        threading.Timer(self.refresh_interval, self.update_status_loop).start()

def main():
    app = QApplication(sys.argv)
    gui = WireGuardGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
