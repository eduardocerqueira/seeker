#date: 2025-02-03T17:05:35Z
#url: https://api.github.com/gists/2e20885c1ec949b01fac371f9ec6af30
#owner: https://api.github.com/users/Alvinislazy

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QPushButton, 
    QFileDialog, QWidget, QVBoxLayout, QHeaderView, QHBoxLayout, QLabel, 
    QLineEdit, QTextEdit, QComboBox, QCheckBox, QSplitter, QMenu, QProgressBar, QTabWidget
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QObject, QThread, QTimer
from PyQt5.QtGui import QFont, QColor
from config import save_config, load_config, save_state, load_state
from utils import fetch_output_folder, open_output_folder
from render_thread import RenderThread
from resource_monitor import ResourceMonitor
import sys
import os
import json
import subprocess
import re

class RenderManagerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SceneWeaver")
        self.setGeometry(100, 100, 1200, 800)

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QHBoxLayout()
        self.main_widget.setLayout(self.layout)

        # Left side layout (Table and Tabs)
        self.left_side_layout = QVBoxLayout()
        self.layout.addLayout(self.left_side_layout)

        # Table setup
        self.table = QTableWidget()
        self.table.setColumnCount(4)  # Columns: File Name, Job Status, Output Folder, Include in Render
        self.table.setHorizontalHeaderLabels(["File Name", "Job Status", "Output Folder", "Include in Render"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)  # Allow column resizing

        # Set column widths as percentages of the table width
        self.table.setColumnWidth(0, int(self.table.width() * 1.50))  # File Name: 40%
        self.table.setColumnWidth(1, int(self.table.width() * 0.40))  # Job Status: 30%
        self.table.setColumnWidth(2, int(self.table.width() * 0.40))  # Output Folder: 20%
        self.table.setColumnWidth(3, int(self.table.width() * 0.40))  # Include in Render: 10%

        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.left_side_layout.addWidget(self.table)

        # Disable scroll wheel for the table
        self.table.wheelEvent = lambda event: event.ignore()

        # Tab Widget for Logs, WeaverSync, and WeaverOutput
        self.tab_widget = QTabWidget()
        self.left_side_layout.addWidget(self.tab_widget)

        # Logs Tab
        self.logs_tab = QWidget()
        self.logs_layout = QVBoxLayout()
        self.logs_tab.setLayout(self.logs_layout)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.logs_layout.addWidget(self.log_display)

        self.tab_widget.addTab(self.logs_tab, "Logs")

        # WeaverSync Tab
        self.setup_weaversync_tab()

        # WeaverOutput Tab
        self.setup_weaveroutput_tab()

        # Right side layout (Resource Monitoring)
        self.right_side_layout = QVBoxLayout()
        self.layout.addLayout(self.right_side_layout)

        # Resource Monitoring Tab
        self.resource_monitor = ResourceMonitor()
        self.right_side_layout.addWidget(self.resource_monitor)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #5E81AC; }")
        self.left_side_layout.addWidget(self.progress_bar)

        # Render Settings
        self.settings_layout = QHBoxLayout()
        self.left_side_layout.addLayout(self.settings_layout)

        # Frame Range Override
        self.frame_override_checkbox = QCheckBox("Override Frame Range")
        self.settings_layout.addWidget(self.frame_override_checkbox)

        self.frame_start_input = QLineEdit("1")
        self.frame_start_input.setPlaceholderText("Start Frame")
        self.frame_start_input.setMaximumWidth(100)
        self.settings_layout.addWidget(self.frame_start_input)

        self.frame_end_input = QLineEdit("100")
        self.frame_end_input.setPlaceholderText("End Frame")
        self.frame_end_input.setMaximumWidth(100)
        self.settings_layout.addWidget(self.frame_end_input)

        # Blender Executable Selection
        self.blender_path_button = QPushButton("Set Blender Executable")
        self.settings_layout.addWidget(self.blender_path_button)
        self.blender_path_button.clicked.connect(self.select_blender_executable)

        # Render All Button
        self.render_all_button = QPushButton("Render All")
        self.left_side_layout.addWidget(self.render_all_button)
        self.render_all_button.clicked.connect(self.toggle_render_all)

        # Blender executable path
        self.blender_executable = ""

        # Load saved Blender executable path and state
        self.load_config()
        self.load_state()

        # Styling
        self.setStyleSheet("""
            QWidget {
                background-color: #2E3440;
                color: #D8DEE9;
                font-family: "Segoe UI";
            }
            QTableWidget {
                gridline-color: #4C566A;
            }
            QHeaderView::section {
                background-color: #4C566A;
                color: #D8DEE9;
                padding: 5px;
            }
            QPushButton {
                background-color: #5E81AC;
                color: #D8DEE9;
                border: none;
                padding: 8px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #81A1C1;
            }
            QLineEdit, QComboBox {
                background-color: #3B4252;
                color: #D8DEE9;
                border: 1px solid #4C566A;
                padding: 5px;
                border-radius: 3px;
            }
            QTextEdit {
                background-color: #3B4252;
                color: #D8DEE9;
                border: 1px solid #4C566A;
                padding: 5px;
                border-radius: 3px;
            }
            QProgressBar {
                background-color: #3B4252;
                color: #D8DEE9;
                border: 1px solid #4C566A;
                padding: 2px;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background-color: #5E81AC;
                border-radius: 2px;
            }
            QProgressBar::chunk:finished {
                background-color: #4CAF50;  /* Green for completed */
            }
            QTabWidget::pane {
                border: 1px solid #4C566A;
                background-color: #3B4252;
            }
            QTabBar::tab {
                background-color: #4C566A;
                color: #D8DEE9;
                padding: 8px;
                border: 1px solid #4C566A;
                border-bottom-color: #3B4252;
            }
            QTabBar::tab:selected {
                background-color: #5E81AC;
                border-bottom-color: #5E81AC;
            }
        """)

        # Enable keyboard shortcuts
        self.table.setFocusPolicy(Qt.StrongFocus)
        self.table.keyPressEvent = self.key_press_event

        # Enable double-click to load files
        self.table.doubleClicked.connect(self.load_files)

        # Enable right-click context menu to load files
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)

        # Dictionary to store active render threads
        self.active_threads = {}

        # Dictionary to store last rendered frame for each file
        self.last_rendered_frame = {}

        # Total frames to render
        self.total_frames = 0
        self.completed_frames = 0

    def setup_weaversync_tab(self):
        """Setup WeaverSync tab with advanced settings."""
        self.weaversync_tab = QWidget()
        self.weaversync_layout = QVBoxLayout()
        self.weaversync_tab.setLayout(self.weaversync_layout)

        # Sub-Tabs for WeaverSync
        self.weaversync_sub_tabs = QTabWidget()
        self.weaversync_layout.addWidget(self.weaversync_sub_tabs)

        # Sub-Tab 1: Settings
        self.settings_tab = QWidget()
        self.settings_layout = QVBoxLayout()
        self.settings_tab.setLayout(self.settings_layout)

        # Resolution Override
        self.resolution_override_checkbox = QCheckBox("Override Resolution")
        self.settings_layout.addWidget(self.resolution_override_checkbox)

        # Resolution Inputs
        self.resolution_layout = QVBoxLayout()  # Change to vertical layout
        self.resolution_width_input = QLineEdit("1920")
        self.resolution_width_input.setPlaceholderText("Width")
        self.resolution_width_input.setMaximumWidth(100)
        self.resolution_layout.addWidget(self.resolution_width_input)

        self.resolution_height_input = QLineEdit("1080")
        self.resolution_height_input.setPlaceholderText("Height")
        self.resolution_height_input.setMaximumWidth(100)
        self.resolution_layout.addWidget(self.resolution_height_input)

        self.settings_layout.addLayout(self.resolution_layout)

        # Render Samples
        self.render_samples_checkbox = QCheckBox("Override Render Samples")
        self.settings_layout.addWidget(self.render_samples_checkbox)

        self.render_samples_input = QLineEdit("128")
        self.render_samples_input.setPlaceholderText("Samples")
        self.render_samples_input.setMaximumWidth(80)  # Reduced width
        self.settings_layout.addWidget(self.render_samples_input)

        self.weaversync_sub_tabs.addTab(self.settings_tab, "SettingsSync")

        # Persistent Data Checkbox
        self.persistent_data_checkbox = QCheckBox("Enable Persistent Data")
        self.settings_layout.addWidget(self.persistent_data_checkbox)

        # Sync Buttons
        self.sync_all_button = QPushButton("Sync it")
        self.sync_all_button.clicked.connect(self.sync_all_files)
        self.settings_layout.addWidget(self.sync_all_button)

        self.sync_selected_button = QPushButton("Sync it for only selected")
        self.sync_selected_button.clicked.connect(self.sync_selected_files)
        self.settings_layout.addWidget(self.sync_selected_button)

        # Sub-Tab 2: Status
        self.status_tab = QWidget()
        self.status_layout = QVBoxLayout()
        self.status_tab.setLayout(self.status_layout)

        self.status_label = QLabel("WeaverSync Status")
        self.status_layout.addWidget(self.status_label)

        self.weaversync_sub_tabs.addTab(self.status_tab, "Status")

        self.tab_widget.addTab(self.weaversync_tab, "WeaverSync")

    def sync_all_files(self):
        """Apply overridden settings to all files."""
        for row in range(self.table.rowCount()):
            self.apply_settings_to_file(row)

    def sync_selected_files(self):
        """Apply overridden settings to selected files."""
        selected_rows = set(index.row() for index in self.table.selectedIndexes())
        for row in selected_rows:
            self.apply_settings_to_file(row)

    def apply_settings_to_file(self, row):
        """Apply overridden settings to a specific file."""
        # Example: Override resolution
        if self.resolution_override_checkbox.isChecked():
            width = self.resolution_width_input.text()
            height = self.resolution_height_input.text()
            # Apply resolution settings to the file (you'll need to implement this logic)
            self.log_display.append(f"Applied resolution {width}x{height} to row {row}.")

    def setup_weaveroutput_tab(self):
        """Setup WeaverOutput tab."""
        self.weaveroutput_tab = QWidget()
        self.weaveroutput_layout = QVBoxLayout()
        self.weaveroutput_tab.setLayout(self.weaveroutput_layout)

        # Enable WeaverOutput Option
        self.weaveroutput_checkbox = QCheckBox("Enable WeaverOutput")
        self.weaveroutput_layout.addWidget(self.weaveroutput_checkbox)

        # Parent Folder Path
        self.parent_folder_label = QLabel("Parent Folder:")
        self.weaveroutput_layout.addWidget(self.parent_folder_label)

        self.parent_folder_input = QLineEdit()
        self.parent_folder_input.setPlaceholderText("Select parent folder for renders")
        self.weaveroutput_layout.addWidget(self.parent_folder_input)

        self.parent_folder_button = QPushButton("Browse")
        self.parent_folder_button.clicked.connect(self.select_parent_folder)
        self.weaveroutput_layout.addWidget(self.parent_folder_button)

        # Output File Type Override
        self.output_override_checkbox = QCheckBox("Override Output File Type")
        self.weaveroutput_layout.addWidget(self.output_override_checkbox)

        self.output_type_combo = QComboBox()
        self.output_type_combo.addItems(["PNG", "JPEG", "EXR", "TIFF", "OPEN_EXR", "FFMPEG"])
        self.weaveroutput_layout.addWidget(self.output_type_combo)

        self.tab_widget.addTab(self.weaveroutput_tab, "WeaverOutput")

    def toggle_render_all(self):
        """Toggle between Render All and Pause Render."""
        if self.render_all_button.text() == "Render All":
            self.render_all_button.setText("Pause Render")
            self.render_all_files()
        else:
            self.render_all_button.setText("Render All")
            self.stop_render()

    def stop_render(self):
        """Stop rendering for all files."""
        for row in self.active_threads:
            self.active_threads[row].stop()  # Stop the thread
            file_name = self.table.item(row, 0).text()
            self.table.item(row, 1).setText("Stopped")  # Update status
            self.table.item(row, 1).setTextAlignment(Qt.AlignCenter)  # Center-align text
            self.log_display.append(f"Rendering stopped for {file_name}.")

    def select_parent_folder(self):
        """Let the user select the parent folder for WeaverOutput."""
        folder = QFileDialog.getExistingDirectory(self, "Select Parent Folder")
        if folder:
            self.parent_folder_input.setText(folder)

    def open_output_folder(self, folder):
        """Open the output folder for a file."""
        if os.path.exists(folder):
            open_output_folder(folder)
        else:
            self.log_display.append(f"Error: Folder {folder} does not exist.")

    def add_file_to_table(self, file):
        """Add a file to the table."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        # File Name
        file_item = QTableWidgetItem(os.path.basename(file))
        file_item.setData(Qt.UserRole, file)  # Store full path
        file_item.setFlags(file_item.flags() ^ Qt.ItemIsEditable)
        self.table.setItem(row, 0, file_item)

        # Job Status (Queued by default)
        status_item = QTableWidgetItem("Queued")
        status_item.setTextAlignment(Qt.AlignCenter)  # Center-align text
        status_item.setFlags(status_item.flags() ^ Qt.ItemIsEditable)
        self.table.setItem(row, 1, status_item)

        # Output Folder
        output_folder = self.get_output_folder(file)
        output_folder_item = QTableWidgetItem(output_folder)
        output_folder_item.setFlags(output_folder_item.flags() ^ Qt.ItemIsEditable)
        self.table.setItem(row, 2, output_folder_item)

        # Include in Render (Checkbox)
        include_item = QTableWidgetItem()
        include_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        include_item.setCheckState(Qt.Checked)  # Default to checked
        self.table.setItem(row, 3, include_item)

        # Output Folder Button
        output_folder_button = QPushButton("Open Folder")
        output_folder_button.clicked.connect(lambda _, path=output_folder: self.open_output_folder(path))
        self.table.setCellWidget(row, 2, output_folder_button)

    def load_files(self):
        """Load files into the table."""
        files, _ = QFileDialog.getOpenFileNames(self, "Select Blender Files", "", "Blender Files (*.blend)")
        if files:
            for file in files:
                self.add_file_to_table(file)

    def get_output_folder(self, file):
        """Get the output folder for a file, considering WeaverOutput settings."""
        if self.weaveroutput_checkbox.isChecked() and self.parent_folder_input.text():
            # Use WeaverOutput parent folder
            parent_folder = self.parent_folder_input.text()
            file_name = os.path.splitext(os.path.basename(file))[0]  # Remove .blend extension
            return os.path.join(parent_folder, file_name)
        else:
            # Use default output folder from the .blend file
            return fetch_output_folder(self.blender_executable, file)

    def key_press_event(self, event):
        """Handle keyboard shortcuts."""
        if event.key() in (Qt.Key_Delete, Qt.Key_X):
            self.delete_selected_file()

    def delete_selected_file(self):
        """Delete the selected file from the table."""
        selected_row = self.table.currentRow()
        if selected_row >= 0:
            self.table.removeRow(selected_row)

    def show_context_menu(self, position):
        """Show a context menu for loading files."""
        menu = QMenu()
        load_files_action = menu.addAction("Load Files")
        load_files_action.triggered.connect(self.load_files)

        # Add "Stop Render" action
        stop_render_action = menu.addAction("Stop Render")
        stop_render_action.triggered.connect(self.stop_render)

        # Add "Restart Render from Beginning" action
        restart_beginning_action = menu.addAction("Restart Render from Beginning")
        restart_beginning_action.triggered.connect(lambda: self.restart_render(from_beginning=True))

        # Add "Restart Render from Last Frame" action
        restart_last_frame_action = menu.addAction("Restart Render from Last Frame")
        restart_last_frame_action.triggered.connect(lambda: self.restart_render(from_beginning=False))

        menu.exec_(self.table.viewport().mapToGlobal(position))

    def restart_render(self, from_beginning=True):
        """Restart rendering for the selected files."""
        selected_rows = set(index.row() for index in self.table.selectedIndexes())
        for row in selected_rows:
            file = self.table.item(row, 0).data(Qt.UserRole)
            self.start_render(file, row, from_beginning=from_beginning)

    def start_render(self, file, row, from_beginning=True):
        """Render a single file."""
        if not self.blender_executable:
            self.select_blender_executable()
            if not self.blender_executable:
                self.log_display.append("Error: Blender executable not set!")
                return

        # Update job status to "In Progress"
        self.table.item(row, 1).setText("In Progress")
        self.table.item(row, 1).setTextAlignment(Qt.AlignCenter)  # Center-align text
        self.table.viewport().update()  # Force UI update

        # Build the Blender command
        command = [self.blender_executable, "-b", file]

        # Add frame range override if enabled
        if self.frame_override_checkbox.isChecked():
            start_frame = int(self.frame_start_input.text())
            end_frame = int(self.frame_end_input.text())

            if not from_beginning and row in self.last_rendered_frame:
                # Resume from the last rendered frame
                start_frame = self.last_rendered_frame[row] + 1

            command.extend(["-s", str(start_frame)])  # Start frame
            command.extend(["-e", str(end_frame)])    # End frame

        # Add output format override if enabled
        if self.output_override_checkbox.isChecked():
            output_format = self.output_type_combo.currentText()
            command.extend(["-F", output_format])  # Output format

        # Add output directory override if WeaverOutput is enabled
        if self.weaveroutput_checkbox.isChecked() and self.parent_folder_input.text():
            parent_folder = self.parent_folder_input.text()
            file_name = os.path.splitext(os.path.basename(file))[0]  # Remove .blend extension
            output_folder = os.path.join(parent_folder, file_name)

            # Create the output directory if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)

            # Override the output directory
            command.extend(["-o", os.path.join(output_folder, "####")])  # Blender's placeholder for frame numbers

        # Add the render command
        command.extend(["-a"])  # Render animation

        # Create and start the render thread
        self.render_thread = RenderThread(command, row, os.path.basename(file))
        self.render_thread.log_signal.connect(self.log_display.append)  # Connect log signal
        self.render_thread.status_signal.connect(lambda status: self.table.item(row, 1).setText(status))  # Connect status signal
        self.render_thread.frame_signal.connect(lambda frame: self.update_frame_progress(row, frame))  # Connect frame progress signal
        self.render_thread.start()  # Start the thread

        # Store the thread in the active_threads dictionary
        self.active_threads[row] = self.render_thread

    def update_frame_progress(self, row, frame):
        """Update the frame progress in the Job Status column."""
        status_item = self.table.item(row, 1)
        if status_item:
            total_frames = int(self.frame_end_input.text()) - int(self.frame_start_input.text()) + 1
            status_item.setText(f"{frame}/{total_frames}")
            status_item.setTextAlignment(Qt.AlignCenter)  # Center-align text

            # Update the last rendered frame
            self.last_rendered_frame[row] = int(frame)

            # Update progress bar
            self.completed_frames += 1
            self.progress_bar.setValue(self.completed_frames)

    def render_all_files(self):
        """Render all files in the table."""
        total_files = self.table.rowCount()
        if total_files == 0:
            self.log_display.append("No files to render.")
            return

        # Calculate total frames to render across all files
        self.total_frames = 0
        for row in range(total_files):
            include_item = self.table.item(row, 3)
            if include_item.checkState() == Qt.Checked:
                start_frame = int(self.frame_start_input.text())
                end_frame = int(self.frame_end_input.text())
                self.total_frames += (end_frame - start_frame + 1)

        self.completed_frames = 0
        self.progress_bar.setMaximum(self.total_frames)
        self.progress_bar.setValue(0)

        for row in range(total_files):
            include_item = self.table.item(row, 3)
            if include_item.checkState() == Qt.Checked:
                file = self.table.item(row, 0).data(Qt.UserRole)
                self.start_render(file, row)
                QApplication.processEvents()  # Update UI

        self.log_display.append("Batch rendering started.")

    def select_blender_executable(self):
        """Let the user select the Blender executable and save it to config."""
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Blender Executable", 
            "", 
            "Executable Files (*.exe);;All Files (*)"
        )
        if path:
            self.blender_executable = path
            self.log_display.append(f"Blender executable set to: {path}")
            save_config(self.blender_executable)  # Save the path to config

    def load_config(self):
        """Load the Blender executable path from a config file."""
        self.blender_executable = load_config()
        if self.blender_executable:
            self.log_display.append(f"Loaded Blender executable from config: {self.blender_executable}")

    def load_state(self):
        """Load the saved state from a file."""
        state = load_state()  # Load state from the config module
        if state:
            # Load Blender executable path
            self.blender_executable = state.get("blender_executable", "")
            
            # Load frame range override settings
            self.frame_override_checkbox.setChecked(state.get("frame_override", False))
            self.frame_start_input.setText(state.get("frame_start", "1"))
            self.frame_end_input.setText(state.get("frame_end", "100"))
            
            # Load output format override settings
            self.output_override_checkbox.setChecked(state.get("output_override", False))
            self.output_type_combo.setCurrentText(state.get("output_type", "PNG"))
            
            # Load WeaverOutput settings
            self.weaveroutput_checkbox.setChecked(state.get("weaveroutput_enabled", False))
            self.parent_folder_input.setText(state.get("parent_folder", ""))
            
            # Load files and their states
            for file_data in state.get("files", []):
                self.add_file_to_table(file_data["file"])
                row = self.table.rowCount() - 1
                self.table.item(row, 1).setText(file_data["status"])
                self.table.item(row, 3).setCheckState(Qt.Checked if file_data["include"] else Qt.Unchecked)

    def save_state(self):
        """Save the current state to a file."""
        state = {
            "blender_executable": self.blender_executable,
            "files": [],
            "frame_override": self.frame_override_checkbox.isChecked(),
            "frame_start": self.frame_start_input.text(),
            "frame_end": self.frame_end_input.text(),
            "output_override": self.output_override_checkbox.isChecked(),
            "output_type": self.output_type_combo.currentText(),
            "weaveroutput_enabled": self.weaveroutput_checkbox.isChecked(),
            "parent_folder": self.parent_folder_input.text(),
        }
        for row in range(self.table.rowCount()):
            file = self.table.item(row, 0).data(Qt.UserRole)
            status = self.table.item(row, 1).text()
            include = self.table.item(row, 3).checkState() == Qt.Checked
            state["files"].append({
                "file": file,
                "status": status,
                "include": include
            })
        save_state(state)  # Save state using the config module

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RenderManagerGUI()
    window.show()
    sys.exit(app.exec_())