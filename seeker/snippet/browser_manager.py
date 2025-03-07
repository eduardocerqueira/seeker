#date: 2025-03-07T17:01:50Z
#url: https://api.github.com/gists/25628c0ce81bdc090be068bf1125ca02
#owner: https://api.github.com/users/huuthan00

import shutil
import os
import json
import datetime

class BackupManager:
    def __init__(self, config_path="config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.profile_path = self.config["chrome_profile_path"]
        self.backup_path = self.config["backup_path"]

    def create_backup(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.backup_path, f"backup_{timestamp}")
        shutil.copytree(self.profile_path, backup_dir)
        print(f"Backup created: {backup_dir}")

    def restore_backup(self, backup_dir):
        shutil.rmtree(self.profile_path)
        shutil.copytree(backup_dir, self.profile_path)
        print(f"Backup restored from: {backup_dir}")

    def list_backups(self):
        if not os.path.exists(self.backup_path):
            return []
        return [d for d in os.listdir(self.backup_path) if os.path.isdir(os.path.join(self.backup_path, d))]