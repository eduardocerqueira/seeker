#date: 2025-02-03T17:05:35Z
#url: https://api.github.com/gists/2e20885c1ec949b01fac371f9ec6af30
#owner: https://api.github.com/users/Alvinislazy

from PyQt5.QtCore import QThread, pyqtSignal
import subprocess
import re

class RenderThread(QThread):
    """Thread to handle Blender rendering."""
    log_signal = pyqtSignal(str)  # Signal to send log messages
    status_signal = pyqtSignal(str)  # Signal to update job status
    frame_signal = pyqtSignal(int)  # Signal to update frame progress

    def __init__(self, command, row, file_name):
        super().__init__()
        self.command = command
        self.row = row
        self.file_name = file_name
        self._is_running = True
        self.process = None

    def run(self):
        """Run the Blender rendering process."""
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",  # Explicitly set encoding to utf-8
                errors="replace"   # Replace invalid characters
            )

            # Monitor process output
            while self.process.poll() is None and self._is_running:
                output = self.process.stdout.readline()
                if output:
                    self.log_signal.emit(output.strip())  # Emit log signal

                    # Parse frame progress from Blender's output
                    frame_match = re.search(r"Fra:(\d+)", output)
                    if frame_match:
                        frame_number = int(frame_match.group(1))
                        self.frame_signal.emit(frame_number)  # Emit frame progress

            # Check for errors
            if not self._is_running:
                self.status_signal.emit("Stopped")  # Emit status signal
                self.log_signal.emit(f"Rendering stopped for {self.file_name}.")
            elif self.process.returncode == 0:
                self.status_signal.emit("Done")  # Emit status signal
                self.log_signal.emit("Rendering completed.")
            else:
                self.status_signal.emit("Failed")  # Emit status signal
                error_message = self.process.stderr.read()
                self.log_signal.emit(f"Error rendering: {error_message}")

        except Exception as e:
            self.status_signal.emit("Failed")  # Emit status signal
            self.log_signal.emit(f"Unexpected error: {e}")

    def stop(self):
        """Stop the rendering process."""
        self._is_running = False
        if self.process:
            self.process.kill()  # Forcefully terminate the process