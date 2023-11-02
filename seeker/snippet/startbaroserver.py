#date: 2023-11-02T16:55:09Z
#url: https://api.github.com/gists/a1ce53ae34aba94ee0512cdd08d0ff73
#owner: https://api.github.com/users/Maselkov

import os
import subprocess
import getpass
import shutil
import requests
import zipfile
import io

user_name = getpass.getuser()

mods_dir = os.path.join(
    "C:\\Users",
    user_name,
    "AppData",
    "Local",
    "Daedalic Entertainment GmbH",
    "Barotrauma",
    "WorkshopMods",
    "Installed",
)

# Change this
server_dir = "C:\\path\\to\\steamapps\\common\\Barotrauma"
steamcmd_dir = "C:\\steamcmd"


app_id = 602960


def run_steamcmd_command(commands):
    steamcmd_path = os.path.join(steamcmd_dir, "steamcmd.exe")
    full_command = f"{steamcmd_path} {' '.join(commands)}"
    subprocess.run(full_command, shell=True)


set_install_dir_command = f"+force_install_dir {server_dir}"
login_command = "+login anonymous"
commands = [set_install_dir_command, login_command]

mod_ids = [
    folder
    for folder in os.listdir(mods_dir)
    if os.path.isdir(os.path.join(mods_dir, folder))
]

for mod_id in mod_ids:
    mod_download_dir = os.path.join(mods_dir, mod_id)
    workshop_update_command = f"+workshop_download_item {app_id} {mod_id}"
    commands.append(workshop_update_command)

commands.append("+quit")

run_steamcmd_command(commands)

for mod_id in mod_ids:
    source_dir = os.path.join(
        server_dir, "steamapps", "workshop", "content", str(app_id), str(mod_id)
    )
    destination_dir = os.path.join(mods_dir, str(mod_id))
    if os.path.exists(source_dir):
        shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

print("All mods are up to date.\n")

repo = "evilfactory/LuaCsForBarotrauma"
file_name = "luacsforbarotrauma_patch_windows_server.zip"

url = f"https://api.github.com/repos/{repo}/releases/latest"
response = requests.get(url)
release = response.json()

for asset in release["assets"]:
    if asset["name"] == file_name:
        download_url = asset["browser_download_url"]
        break

response = requests.get(download_url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))
zip_file.extractall("C:/ssdsteam/steamapps/common/Barotrauma")

dedicated_server_path = os.path.join(server_dir, "DedicatedServer.exe")
subprocess.Popen([dedicated_server_path])
