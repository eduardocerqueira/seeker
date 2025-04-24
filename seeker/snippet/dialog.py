#date: 2025-04-24T16:57:38Z
#url: https://api.github.com/gists/7aa17b87988694efedfb7c692b7b052a
#owner: https://api.github.com/users/junmaodd

#!/Users/jun.mao/Dropbox/Code/DevTools/.venv/bin/python3
import sys
import argparse
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QComboBox,
    QLabel,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QPushButton,
)


class DialogApp(QMainWindow):
    def __init__(self, title, action, input_content=None, dropdown_options=None):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(800, 400)
        self.action = action
        self.input_content = input_content
        self.dropdown_options = dropdown_options
        self.result = None

        # Call the appropriate dialog based on the action
        if action == "message":
            self.show_message_dialog()
        elif action == "input":
            self.show_multiline_input()
        elif action == "dropdown":
            self.show_dropdown()
        else:
            raise ValueError(f"Unknown action: {action}")

    def show_message_dialog(self):
        """Show a message dialog using input_content as the message."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        label = QTextEdit()
        label.setReadOnly(True)
        label.setMarkdown(self.input_content or "No message provided.")
        layout.addWidget(label)

        central_widget.setLayout(layout)

    def show_multiline_input(self):
        """Show a multiline input dialog initialized with the given content and print the result."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        label = QLabel("Enter your text below:")
        layout.addWidget(label)

        central_widget.setLayout(layout)
        input_area = QTextEdit()
        input_area.setPlainText(self.input_content or "")
        layout.addWidget(input_area)

        # Add an Okay button
        okay_button = QPushButton("Okay")
        okay_button.clicked.connect(lambda: self.set_result(input_area.toPlainText()))
        okay_button.clicked.connect(self.close)
        layout.addWidget(okay_button)

    def set_result(self, result):
        """Set the result attribute."""
        self.result = result

    def show_dropdown(self):
        """Show a dropdown menu (ComboBox), allow selection, and print the selected option."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        label = QLabel("Select an option:")
        layout.addWidget(label)

        dropdown = QComboBox()
        dropdown.addItems(self.dropdown_options or ["Option 1", "Option 2"])
        layout.addWidget(dropdown)

        # Show the selected option and print it to the console
        result_label = QLabel("")
        dropdown.currentIndexChanged.connect(
            lambda: result_label.setText(f"Selected: {dropdown.currentText()}")
        )
        layout.addWidget(result_label)

        # Add an Okay button
        okay_button = QPushButton("Okay")
        okay_button.clicked.connect(lambda: self.set_result(dropdown.currentText()))
        okay_button.clicked.connect(self.close)
        layout.addWidget(okay_button)

        central_widget.setLayout(layout)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="A GUI application with dialog options."
    )
    parser.add_argument(
        "-t",
        "--title",
        type=str,
        default="Dialog Application",
        help="The title of the application window.",
    )
    parser.add_argument(
        "-a",
        "--action",
        type=str,
        choices=["message", "input", "dropdown"],
        default="message",
        help="The action to perform: 'message', 'input', or 'dropdown'.",
    )
    parser.add_argument(
        "-i",
        "--input-content",
        type=str,
        default="",
        help="The content for the message or input dialog.",
    )
    parser.add_argument(
        "-d",
        "--dropdown-options",
        type=str,
        nargs="+",
        default=["Option 1", "Option 2"],
        help="Space-separated options for the dropdown menu.",
    )
    return parser.parse_args()


def main():
    """Main function to run the application."""
    args = parse_args()

    app = QApplication(sys.argv)
    window = DialogApp(
        title=args.title,
        action=args.action,
        input_content=args.input_content,
        dropdown_options=args.dropdown_options,
    )
    window.show()
    exit_code = app.exec_()
    if window.result:
        print(window.result)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
