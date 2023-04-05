#date: 2023-04-05T17:06:35Z
#url: https://api.github.com/gists/cb1a579acfb57b38c7afcc0f5fab437d
#owner: https://api.github.com/users/xmif

import obspython as obs
import shutil
from os import path

def script_description():
    return "Copy recordings to a new directory upon completion."

def script_load(settings):
    global obsSettings;
    obsSettings = settings
    obs.obs_frontend_add_event_callback(on_event)
    print("Script Loaded")

def script_properties():
    props = obs.obs_properties_create()
    obs.obs_properties_add_path(props, "google_drive_path", "Google Drive Path :", obs.OBS_PATH_DIRECTORY, "", "")
    return props

def on_event(event):
    global obsSettings;
    if event == obs.OBS_FRONTEND_EVENT_RECORDING_STOPPED:
        recording_path = obs.obs_frontend_get_last_recording();
        google_drive_path = obs.obs_data_get_string(obsSettings, "google_drive_path")
        shutil.copy2(recording_path, path.join(google_drive_path, path.basename(recording_path)))