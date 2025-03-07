#date: 2025-03-07T17:01:50Z
#url: https://api.github.com/gists/25628c0ce81bdc090be068bf1125ca02
#owner: https://api.github.com/users/huuthan00

import importlib.util
import os
import json

class ScriptManager:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.script_path = self.config["script_path"]

    def run_script(self, driver, script_name):
        script_file = os.path.join(self.script_path, f"{script_name}.py")
        if not os.path.exists(script_file):
            print(f"Script not found: {script_name}")
            return

        spec = importlib.util.spec_from_file_location(script_name, script_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "run"):
            module.run(driver)
        else:
            print(f"Script {script_name} does not have a 'run' function.")