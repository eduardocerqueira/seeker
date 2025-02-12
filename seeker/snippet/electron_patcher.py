#date: 2025-02-12T16:56:39Z
#url: https://api.github.com/gists/7373d551ebb11c38a6f7cb8702998dc3
#owner: https://api.github.com/users/matthew-mclaren

"""
electron_patcher.py: Enforce 'use-angle@1' in Chrome and Electron applications

Version 1.0.1 (2025-02-12)
"""

import enum
import json

from pathlib import Path


class ChromiumSettingsPatcher:

    class AngleVariant(enum.Enum):
        Default = "0"
        OpenGL = "1"
        Metal = "2"

    def __init__(self, state_file: str) -> None:
        self._local_state_file = Path(state_file).expanduser()


    def patch(self) -> None:
        """
        Ensure 'use-angle@1' is set in Chrome's experimental settings
        """
        _desired_key   = "use-angle"
        _desired_value = self.AngleVariant.OpenGL.value

        if not self._local_state_file.exists():
            print("  Local State missing, creating...")
            self._local_state_file.parent.mkdir(parents=True, exist_ok=True)
            state_data = {}
        else:
            print("  Parsing Local State file")
            state_data = json.loads(self._local_state_file.read_bytes())


        if "browser" not in state_data:
            state_data["browser"] = {}
        if "enabled_labs_experiments" not in state_data["browser"]:
            state_data["browser"]["enabled_labs_experiments"] = []

        for key in state_data["browser"]["enabled_labs_experiments"]:
            if "@" not in key:
                continue

            key_pair = key.split("@")
            if len(key_pair) < 2:
                continue
            if key_pair[0] != _desired_key:
                continue
            if key_pair[1] == _desired_value:
                print(f"  {_desired_key}@{_desired_value} is already set")
                break

            index = state_data["browser"]["enabled_labs_experiments"].index(key)
            state_data["browser"]["enabled_labs_experiments"][index] = f"{_desired_key}@{_desired_value}"
            print(f"  Updated {_desired_key}@{_desired_value}")

        if f"{_desired_key}@{_desired_value}" not in state_data["browser"]["enabled_labs_experiments"]:
            state_data["browser"]["enabled_labs_experiments"].append(f"{_desired_key}@{_desired_value}")
            print(f"  Added {_desired_key}@{_desired_value}")

        print("  Writing to Local State file")
        self._local_state_file.write_text(json.dumps(state_data, indent=4))


def main():
    # Patch all Electron applications
    for directory in Path("~/Library/Application Support").expanduser().iterdir():
        if not directory.is_dir():
            continue

        state_file = directory / "Local State"
        if not state_file.exists():
            continue

        print(f"Patching {directory.name}")
        patcher = ChromiumSettingsPatcher(state_file)
        patcher.patch()

    # Patch all Chrome variants
    if Path("~/Library/Application Support/Google").expanduser().exists():
        for directory in Path("~/Library/Application Support/Google").expanduser().iterdir():
            if not directory.is_dir():
                continue

            state_file = directory / "Local State"
            if not state_file.exists():
                continue

            print(f"Patching {directory.name}")
            patcher = ChromiumSettingsPatcher(state_file)
            patcher.patch()

    # Patch Microsoft Teams
    teams_path = Path(
        "~/Library/Containers/com.microsoft.teams2/Data/Library/Application Support/Microsoft/MSTeams/EBWebView/Local State"
    ).expanduser()

    if teams_path.exists():
        print("Patching Microsoft Teams")
        patcher = ChromiumSettingsPatcher(teams_path)
        patcher.patch()


if __name__ == "__main__":
    main()