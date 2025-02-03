#date: 2025-02-03T17:05:35Z
#url: https://api.github.com/gists/2e20885c1ec949b01fac371f9ec6af30
#owner: https://api.github.com/users/Alvinislazy

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import QTimer
import psutil
import pynvml

class ResourceMonitor(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.start_monitoring()

    def setup_ui(self):
        """Setup the resource monitoring UI."""
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # CPU Usage
        self.cpu_label = QLabel("CPU Usage: 0%")
        self.layout.addWidget(self.cpu_label)

        # RAM Usage
        self.ram_label = QLabel("RAM Usage: 0%")
        self.layout.addWidget(self.ram_label)

        # GPU Usage
        self.gpu_label = QLabel("GPU Usage: N/A")
        self.layout.addWidget(self.gpu_label)

        # GPU VRAM Usage
        self.gpu_vram_label = QLabel("GPU VRAM Usage: N/A")
        self.layout.addWidget(self.gpu_vram_label)

    def start_monitoring(self):
        """Start monitoring system resources."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_resource_usage)
        self.timer.start(1000)  # Update every second

    def update_resource_usage(self):
        """Update resource usage labels."""
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        self.cpu_label.setText(f"CPU Usage: {cpu_usage}%")
        self.ram_label.setText(f"RAM Usage: {ram_usage}%")

        # GPU usage (requires additional libraries like `pynvml`)
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_vram_usage = (gpu_memory.used / gpu_memory.total) * 100
            self.gpu_label.setText(f"GPU Usage: {gpu_usage}%")
            self.gpu_vram_label.setText(f"GPU VRAM Usage: {gpu_vram_usage:.1f}%")
        except ImportError:
            self.gpu_label.setText("GPU Usage: N/A")
            self.gpu_vram_label.setText("GPU VRAM Usage: N/A")