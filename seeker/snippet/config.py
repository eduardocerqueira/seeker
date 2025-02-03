#date: 2025-02-03T17:05:35Z
#url: https://api.github.com/gists/2e20885c1ec949b01fac371f9ec6af30
#owner: https://api.github.com/users/Alvinislazy

import json
import os

CONFIG_FILE = "config.json"
STATE_FILE = "state.json"

def load_config():
    """Load the Blender executable path from the config file."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f).get("blender_executable", "")
    return ""

def save_config(blender_executable):
    """Save the Blender executable path to the config file."""
    with open(CONFIG_FILE, "w") as f:
        json.dump({"blender_executable": blender_executable}, f)

def load_state():
    """Load the application state from the state file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_state(state):
    """Save the application state to the state file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)