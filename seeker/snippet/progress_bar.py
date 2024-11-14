#date: 2024-11-14T16:54:06Z
#url: https://api.github.com/gists/478a75e647fcea242e424e4e694284dd
#owner: https://api.github.com/users/joewoz

import sys

class ProgressBar:
    def __init__(self, total_size, width=80):
        self.total_size = total_size
        self.width = width
        self.progress = 0
        self.prefix = "Progress"
        self.suffix = "Complete"
        self.fill = "█"

    def update(self, value: int):
        self.progress = value
        percentage = ("{0:.2f}").format((self.progress / self.total_size) * 100)
        fill_length = int(self.width * self.progress // self.total_size)
        bar = self.fill * fill_length + "-" * (self.width - fill_length)
        sys.stdout.write(f"\r{self.prefix} |{bar}| {percentage}% {self.suffix}")
        sys.stdout.flush()
        if percentage == "100.00":
            self.complete()

    def complete(self):
        sys.stdout.write("d\n")
        sys.stdout.flush()