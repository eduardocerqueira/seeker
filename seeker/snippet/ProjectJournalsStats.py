#date: 2024-11-14T17:11:47Z
#url: https://api.github.com/gists/9909d2478348d611429ee30aeb3b8342
#owner: https://api.github.com/users/Nircek

import random
import time
import threading
from pathlib import Path
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import Tk, Text

DIRECTORY_PATH = "/dmz/@projects/@project-journals"
DISK_SCAN_INTERVAL = 120
WORST_FILES_COUNT = 10


class ProjectJournalsStats:
    def __init__(self, root):
        self.root = root
        self.root.title("Project Journals Stats")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure((0, 1), weight=1, uniform="column")

        self.fig_pie = Figure(figsize=(4, 4), dpi=100)
        self.fig_bar = Figure(figsize=(4, 4), dpi=100)

        self.canvas_pie = FigureCanvasTkAgg(self.fig_pie, master=self.root)
        self.canvas_pie.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, master=self.root)
        self.canvas_bar.get_tk_widget().grid(row=0, column=1, sticky="nsew", rowspan=2)

        self.text_widget = Text(self.root, height=10)
        self.text_widget.grid(row=1, column=0, sticky="nsew")

        self.data = None
        self.gather_data_thread = threading.Thread(target=self.gather_data)
        self.gather_data_thread.daemon = True
        self.gather_data_thread.start()

        self.last_configure_after_id = None
        root.bind("<Configure>", self.on_configure)
        self.root.eval("tk::PlaceWindow . center")

    def on_configure(self, _event=None):
        if self.last_configure_after_id is not None:
            self.root.after_cancel(self.last_configure_after_id)
        self.last_configure_after_id = self.root.after(1000, self.redraw)

    def gather_data(self):
        while True:
            self.data = self.scan_directory()
            self.redraw()
            time.sleep(DISK_SCAN_INTERVAL)

    def scan_directory(self):
        file_paths = [
            (file_path.stat().st_size, file_path.name)
            for file_path in Path(DIRECTORY_PATH).iterdir()
            if file_path.is_file()
        ]
        file_paths.sort()
        file_sizes = [path[0] for path in file_paths]
        empty_files = file_sizes.count(0)
        non_empty_files = len(file_sizes) - empty_files
        file_sizes = [x for x in file_sizes if x != 0]
        threshold = (
            file_paths[WORST_FILES_COUNT][0]
            if len(file_paths) > WORST_FILES_COUNT
            else 0
        )
        file_paths = [(size, name) for size, name in file_paths if size <= threshold]
        random.shuffle(file_paths)
        file_paths = sorted(file_paths[:WORST_FILES_COUNT])
        return empty_files, non_empty_files, file_sizes, file_paths

    def redraw(self, _event=None):
        while self.data is None:
            time.sleep(0.01)
        empty_files, non_empty_files, file_sizes, file_paths = self.data

        self.fig_pie.clear()
        ax_pie = self.fig_pie.add_subplot(111)
        counts = [empty_files, non_empty_files]
        ax_pie.pie(
            counts,
            labels=["Empty journals", "Filled journals"],
            autopct=lambda pct: f"{pct:.0f}% ({int(round(pct/100*sum(counts)))})",
            colors=["orangered", "springgreen"],
        )
        ax_pie.set_title("Journal Status", pad=20)

        self.fig_pie.tight_layout()
        self.canvas_pie.draw()

        self.fig_bar.clear()
        ax_bar = self.fig_bar.add_subplot(111)

        counts, bins = np.histogram(np.log(file_sizes))
        bins = np.exp(bins)
        bucket_labels = [
            f"{bins[i]:.0f}\N{DIVISION SIGN}{bins[i+1]:.0f}"
            for i in range(len(bins) - 1)
        ]

        ax_bar.bar(bucket_labels, counts, color="mediumblue")
        ax_bar.set_xlabel("Journal size in bytes")
        ax_bar.set_ylabel("Count of journals")
        ax_bar.set_title("Distribution of journal sizes", pad=20)
        ax_bar.set_xticks(bucket_labels, bucket_labels, rotation=45, ha="right")

        self.fig_bar.tight_layout()
        self.canvas_bar.draw()

        self.text_widget.delete(1.0, "end")
        for _size, name in file_paths:
            self.text_widget.insert("end", f"{name}\n")


if __name__ == "__main__":
    root = Tk()
    app = ProjectJournalsStats(root)
    root.mainloop()
