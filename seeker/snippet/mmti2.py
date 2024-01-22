#date: 2024-01-22T17:00:05Z
#url: https://api.github.com/gists/24251fc057df7b51d20379d8fa1ecec8
#owner: https://api.github.com/users/tlambert03

import urllib.request
import os

# ------------- make sure we have pymmcore_plus installed ----------------
try:
    import pymmcore_plus

    print("pymmcore_plus found.")
except ImportError:
    # ask to install it
    if input("pymmcore_plus not found. Install? [y/n]") == "y":
        import subprocess

        subprocess.call(["pip", "install", "pymmcore_plus"])
    else:
        raise ImportError(" Please install pymmcore-plus manually or via pip.")

# ------------- make sure we have micro-manager installed ----------------

from pymmcore_plus import find_micromanager

path = find_micromanager()
if path:
    print("Found micromanager at: ", path)
else:
    subprocess.check_output(["mmcore", "intsall"])
    path = find_micromanager()
    if not path:
        raise ImportError("Could not find micromanager. Please install it manually.")
    print("Installed micromanager at: ", path)


# ------------- make sure we have the Ti2_Mic_Driver.dll ----------------

destination = os.path.join(path, "Ti2_Mic_Driver.dll")
if not os.path.exists(destination):
    # download URL to path
    print("Downloading Ti2_Mic_Driver.dll to: ", destination)
    URL = "https://www.dropbox.com/scl/fi/9yd75dhgvg19hs3v7qdso/Ti2_Mic_Driver.dll?rlkey=9x1zqvsoq8ojb5p6uc573h17l&dl=1"
    with urllib.request.urlopen(URL) as response, open(destination, "wb") as out_file:
        out_file.write(response.read())
        assert os.path.exists(destination)
else:
    print("Ti2_Mic_Driver.dll already exists at: ", destination)

# ------------- make sure Ti2 works ----------------

from pymmcore_plus import CMMCorePlus

mmc = CMMCorePlus()
avail = mmc.getAvailableDevices("NikonTi2")
print("Available Ti2 devices: ", avail)
