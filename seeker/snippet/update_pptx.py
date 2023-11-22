#date: 2023-11-22T16:48:44Z
#url: https://api.github.com/gists/3ee0e09e69051e5425e7a5e213f8da94
#owner: https://api.github.com/users/MathItYT

from win32com.client.gencache import EnsureDispatch
import random
import time


powerpoint = EnsureDispatch("PowerPoint.Application")

selected = powerpoint.ActiveWindow.Selection.ShapeRange(1)

try:
    while True:
        selected.TextFrame.TextRange.Text = f"{random.randint(10, 99)}"
        time.sleep(2)
except KeyboardInterrupt:
    print("Exiting...")
